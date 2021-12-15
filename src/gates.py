import torch
import torch.distributed as dist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.utils.checkpoint as checkpoint
import sparse
import sys
import os
import socket
import util
import construct_model


def hard_aggregate(V, V_up, lower=True):
    '''
    Routine for Hard Max and Hard Min aggregation.
            Parameters:
                    V  (Tensor): First operand of the aggregation.
                    V_up (Tensor): Second operand of the aggregation.
                    lower (bool): boolean to select hard max (max) (if true) or hard min (min) (if false)
            Returns:
                    max/min aggregated tensor of V and V_up.
    '''
    if(lower == True): result = torch.max(V, V_up)
    else: result = torch.min(V, V_up)

    return result



def aggregate(V, V_up, lower=True):
    '''
    Routine for Smooth Max and Smooth Min aggregation.
            Parameters:
                    V  (Tensor): First operand of the aggregation.
                    V_up (Tensor): Second operand of the aggregation.
                    lower (bool): boolean to select smooth max (smax) (if true) or smooth min (smin) (if false)
            Returns:
                    smooth max/min aggregated tensor of V and V_up.

    Formula :
                  If lower = true, the routine returns smax(V, V_up) which is a tensor 
                  whose entry i is computed as 
                  smax(V, V_up)[i] = (v[i]*e^(a*v[i]) + v_up[i]*e^(a*v_up[i]))/(e^(a*v[i])+e^(a*v_up[i]))
                  where a is configuration parameter smooth_alpha.

                  If lower = false, the routine returns smin(V, V_up) which is equal to 1 - smax(1-V, 1-V_up)

    '''
    # Read smooth_alpha (exponentiation coefficient) from config json
    smooth_alpha = util.gen_data.config["smooth_alpha"]

    # set W = V or 1-V;  W_up=V_up or 1-V_up depending on whether we compute smooth max or smooth min
    if(lower==True):
        W = V
        W_up = V_up
    else:
        W = 1 - V
        W_up = 1 - V_up

    # Apply the above mentioned formula to compute W_proofs which is smooth max of W and W_up
    W_exp = torch.exp(smooth_alpha*W)
    W_up_exp = torch.exp(smooth_alpha*W_up)
    W_proofs = (W_up * W_up_exp + W * W_exp) / (W_up_exp + W_exp)

    # Return W_proofs or 1-W_proofs depending on whether we compute smooth max or smooth min
    if(lower==True):
        return W_proofs
    else:
        return 1-W_proofs

def gen_downward(G, contr, org_bound, orphans, lu_flag):
    '''
    Routine for doing aggregation of contributions of bound in downward inference.
            Parameters:
                    G  (SparseTensor): Sparse Tensor defining connectivity of nodes.
                    contr (Tensor): Tensor of dimension G._nnz(). 
                                    contr[i] = bound update contributed by node G._indices()[0][i] to G._indices()[1][i].
                    org_bound (Tensor): Tensor of dimension G.size()[0]. original bound of nodes before downward inference.
                    orphans (Tensor):  0/1 Tensor of dimension G.size()[0]. 
                                    orphan[j] = 1 if j is a leaf node which means G._indices()[1] does not contain j.
                    lu_flag (bool): boolean to select smooth max (if true) or smooth min (if false) for aggregation
            Returns:
                    smooth max/min aggregated updated bounds of nodes.
                  
    
    Formula: 
                  Let x1, x2, ..., xn be the bound update contributions to a node c from its parents p1,p2,...pn respectively.
                  Let o be the original bound of node c.
                  
                  The updated bound u of node c is computed in two steps 
                  (i) the aggregate of the parents' contributions is done following
                         agg = \sum_i=1^n{xi * e^(a*(xi))} / \sum_i=1^n{e^(a*(xi))}

                  (ii) smooth max / min of agg and o is computed depending on whether lu_flag is True/False.
                             
                         u = smax(o, agg) if lu_flag is True
                           = 1 - smax(1-o,1-agg) else.


     Steps:
                  1. Read smooth_alpha a from config json
                  2. Define two SparseTensors G1 and G2 with the same nnz pattern as G. The values in G1 is set 
                     as contr * e^(a* contr) and those in G2 is set as  e^(a* contr).

                  3. Do a column sum on G1 and G2. This will aggregate for every node the contributions
                     (in proper form) from all its parents. 
                     The column sum of G1 and G2 will return respectively the numerator and denominator of the RHS in (i)
        
                  4. Return the aggregate of original bound and the updates obtained above.

                  Note: For leaf nodes, both the numerator and denominators obtained in step 3 will be 0. To avoid this  
                  boundary case, the denominator is set to 1 for such nodes. Thus denominator should be replaced by
                  'denominator + orphan' before doing the division. Also we should not do any aggregation 
                  and return the original bound for such nodes. 

    '''

    # read smooth_alpha a from config json
    smooth_alpha = util.gen_data.config["smooth_alpha"]

    # define two SparseTensors with the same nnz pattern as G, with values contr * e^(a* contr) and e^(a* contr).
    # and do a column sum on these SparseTensors to get numer and denom respectively.

    contr_exp = torch.exp(smooth_alpha * contr) #size m
    numer = sparse.ComputeSparseColSum.apply(G, contr * contr_exp)
    denom = sparse.ComputeSparseColSum.apply(G, contr_exp) #size n

    # for boundary case of leaf nodes which do not have any parent (i.e., nodes for which orphan =1) denom = 0. 
    # to avoid division by zero, denom is set to 1 for such nodes.
    denom = denom + orphans

    # Aggregate numer/denom with original bound org_bound using routine aggregate.
    if(lu_flag == True):
        agg = aggregate(org_bound, numer/denom, lower=lu_flag)
    else:     
        agg = aggregate(org_bound, 1 - (numer/denom), lower=lu_flag)

    # for orphan nodes, we return the org_bound;  for other nodes we return the aggregated bound.
    new_bound = agg * (~orphans) + org_bound * orphans
    return(new_bound)

def val_clamp_grad(x, _min: float = 0, _max: float = 1) -> torch.FloatTensor:
    return x.clamp(_min, _max)

def val_clamp(x, _min: float = 0, _max: float = 1) -> torch.FloatTensor:
    """gradient-transparent clamping to clamp values between [min, max]"""
    clamp_min = (x.detach() - _min).clamp(max=0)
    clamp_max = (x.detach() - _max).clamp(min=0)
    return x - clamp_max - clamp_min

class AndNet:
    '''
    Class to define and operate on the And Net of the graph. 
    Salient class members:
                    G (SparseTensor): Sparse Tensor defining connectivity of nodes in the And Net.
                    wptr (Tensor): Tensor of dimension  G._nnz() specifying the mapping of And Net edges to gLNN.
                    bptr (Tensor): Tensor of dimension  G.size()[0] specifying the mapping of And Net nodes to gLNN. 
                    W (Tensor): Tensor of dimension  G._nnz() specifying the edge weights. 
                    invW (Tensor): Tensor of dimension  G._nnz() specifying the inverse of the edge weights. 
                    Gw (SparseTensor): G with non zero values set to W. 
                    GwTr (SparseTensor): Transpose of Gw. 
                    B (SparseTensor): Tensor of dimension  G.size()[0] specifying the node bias.
                    down_orphans (Boolean Tensor): True for nodes which do not have any child.
                    up_orphans (Boolean Tensor): True for nodes which do not have any parent.
    '''
    def __init__(self, G, wptr, bptr, squeeze=True):
        '''
         Routine for initializing the class members.
            Parameters:
                    G (SparseTensor): Sparse Tensor defining connectivity of nodes in the And Net.
                    wptr (Tensor): Tensor of dimension  G._nnz() specifying the mapping of And Net edges to gLNN.
                    bptr (Tensor): Tensor of dimension  G.size()[0] specifying the mapping of And Net nodes to gLNN. 
    
        '''
        assert(G.is_coalesced()) 
        self.G = G
        self.n = G.size()[0]
        self.nnz = G._nnz()
        assert(self.nnz == wptr.shape[0])
        assert(bptr.nelement() == self.n)
        self.wptr = wptr 
        self.bptr = bptr
        #
        self.W = None
        self.invW = None
        self.Gw = None
        self.GwTr = None
        self.B = None

        fan_out = torch.sparse.sum(self.G, 0).to_dense()
        if(squeeze == True):
            self.down_orphans = (fan_out == 0).unsqueeze(1)
        else:
            self.down_orphans = (fan_out == 0)

        fan_in = torch.sparse.sum(self.G, 1).to_dense()
        if(squeeze == True):
            self.up_orphans = (fan_in == 0).unsqueeze(1)
        else:
            self.up_orphans = (fan_in == 0)

    def to_device(self, device):
        '''
         Routine to move class elements to cpu/gpu.
            Parameters:
                    device : CPU or particular GPU 
    
        '''
        self.G = self.G.to(device)
        self.wptr = self.wptr.to(device)
        self.bptr = self.bptr.to(device)
        self.down_orphans = self.down_orphans.to(device)
        self.up_orphans = self.up_orphans.to(device)
        self.W = None
        self.invW = None
        self.Gw = None
        self.GwTr = None
        self.B = None

    def setup_weights(self, gW, gB, squeeze=True):
        '''
         Routine to set up  weights and bias 
            Parameters:
                    gW (Tensor): Tensor of dim gLNN._nnz() specifying weight values
                                 (includes both learnable and non-learnable parameters)
                    gB(Tensor): Tensor of dim gLNN.size()[0] specifying bias parameters
    
        '''
        self.W = gW[self.wptr]
        self.invW = 1/self.W
        self.Gw = sparse.CreateGw.apply(self.G, self.W)
        self.GwTr = self.Gw.transpose(0,1).coalesce()
        if(squeeze==True):
            self.B = gB[self.bptr].unsqueeze(1)
        else:
            self.B = gB[self.bptr]

    def upward(self, L, U, clamp=val_clamp):
        '''
         Routine for upward inference of And Net.
            Parameters:
                    L (Tensor): Lower Bound Tensor of nodes.
                    U (Tensor): Upper Bound Tensor of nodes.
   
            Formula:
                    L_up = B - Gw x (1 - L) 
                    U_up = B - Gw x (1 - U)
                    clamp L_up, U_up in [0,1]
                    return smax(L, L_up), smin(U,U_up) 
        '''

        # Lt = Gw * (1 - L) 
        Lt = sparse.ComputeSparseMv(self.Gw, self.GwTr, 1-L)

        L_up = self.B - Lt
        L_up = clamp(L_up, 0, 1)

        # Ut = Gw * (1 - U)
        Ut = sparse.ComputeSparseMv(self.Gw, self.GwTr, 1-U)

        U_up = self.B - Ut
        U_up = clamp(U_up, 0, 1)
        Lnew = self.up_orphans * L + (~self.up_orphans) * aggregate(L, L_up, lower=True)
        Unew = self.up_orphans * U + (~self.up_orphans) * aggregate(U, U_up, lower=False)
        return Lnew, Unew

    def downward(self, L, U, squeeze=True, clamp=val_clamp):
        '''
         Routine for downward inference of And Net.
            Parameters:
                    L (Tensor): Lower Bound Tensor of nodes.
                    U (Tensor): Upper Bound Tensor of nodes.

            Formula:
                    Recall the formula of upward which can be represented for a particular parent p as:
                         V_up[p] = B[p] - \sum_c|{c is child of p} Gw[p][c] * (1 - V[c]) 
                             where V can be either L or U.
                    Thus for a particular child c' the downward contribution from parent p is,
                        Gw[p][c'] * (1 - V_contri[p,c'])  = B[p] - \sum_{c\c'} Gw[p][c](1-V[c]) - V[p]  
                      => 1 - V_contri[p,c'] = (1/Gw[p][c'])[B[p] - V[p] - \sum_{c\c'} Gw[p][c](1-V[c])]
                      => V_contri[p,c'] = 1 - (1/Gw[p][c'])[B[p] - V[p] - \sum_{c\c'} Gw[p][c](1-V[c])]

                    Therefore,
                         L_contri[p,c'] = 1 - (1/Gw[p][c'])[B[p] - L[p] - \sum_{c\c'} Gw[p][c](1-U[c])]
                                        = 1 - (1/Gw[p][c'])[B[p] - L[p] - \sum_c Gw[p][c](1-U[c]) + Gw[p][c'](1-U[c'])]
                                        = U[c'] - (1/Gw[p][c'])[B[p] - L[p] - \sum_c Gw[p][c](1-U[c])]

                    and similarly,
                         U_contri[p,c'] = 1 - (1/Gw[p][c'])[B[p] - U[p] - \sum_{c\c'} Gw[p][c](1-L[c])]
                                        = L[c'] - (1/Gw[p][c'])[B[p] - U[p] - \sum_c Gw[p][c](1-L[c])]


                    The parent values L[p] and U[p] are actually unclamped before applying the above equations. 
                    By unclamping, the values set to 0 in L  or 1 in U are brought back to range outside [0,1].
                    As before the upward equation is: L_up = B - Gw x (1 - L)  
                    If Lup[p] <= 0, Lunc[p] = (B - Gw x 1)[p]  i.e., unclamped to the max negative value the entry can be.
                    If Uup[p] >=1, Uunc[p] = B[p] i.e., unclamped to the max positive value the entry can be.


                    Thus the final form of the above equations become:
                         L_contri[p,c'] = U[c'] - (1/Gw[p][c'])[B[p] - Lunc[p] - \sum_c Gw[p][c](1-U[c])]
                         U_contri[p,c'] = L[c'] - (1/Gw[p][c'])[B[p] - Uunc[p] - \sum_c Gw[p][c](1-L[c])]

                    Once we have the downward contributions of each parent-child pair,
                    We then need to aggregate contributions from all parents for a child.
 
           Steps:

                    1. Unclamp L and U.
                       Thus if L[p]=0, assign Lunc[p] = (B - Gw x 1)[p]; else maintain Lunc[p] =  L[p].
                       and if U[p] = 1, assign Uunc[p] = B[p]; else maintain Uunc[p] =  U[p].
                       
                    2. Compute L_sum[p] = [B[p] - Lunc[p] - \sum_c Gw[p][c](1-U[c])] 
                       and L_contri[p,c'] = U[c'] - (1/Gw[p][c'])L_sum[p].
                       Clamp L_contri to range [0,1].

                    3. Do the aggregate of L contributions from different parents using gen_downward.

                    4. Compute U_sum[p] = [B[p] - Uunc[p] - \sum_c Gw[p][c](1-L[c])] 
                       and U_contri[p,c'] = L[c'] - (1/Gw[p][c'])U_sum[p].
                       Clamp U_contri to range [0,1].

                    5. Do the aggregate of U contributions from different parents using gen_downward.
        '''

        # Lt = Gw * (1 - L) 
        par_ids = self.G._indices()[0]
        child_ids = self.G._indices()[1]

        # unclamp L
        if(squeeze==True):
            A1 = sparse.ComputeSparseRowSum.apply(self.G, self.W).unsqueeze(1)
        else:
            A1 = sparse.ComputeSparseRowSum.apply(self.G, self.W)

        L_unclamped = L + (self.B - A1) * (L <= 0).float()

        # compute L_sum and L_contri using the formulas defined above
        L_sum = self.B - L_unclamped - sparse.ComputeSparseMv(self.Gw, self.GwTr, 1-U)
        if(squeeze==True):
            Lcontr = U[child_ids] - self.invW.unsqueeze(1) * L_sum[par_ids]
        else:
            Lcontr = U[child_ids] - self.invW * L_sum[par_ids] #size m

        # clamp L_contri to range [0,1].
        Lcontr = clamp(Lcontr, 0, 1)

        # aggregate L contributions from different parents using gen_downward.
        L = gen_downward(self.G, Lcontr, L, self.down_orphans, True)

        ######################

        # unclamp U
        U_unclamped = U + (self.B - 1) * (U >= 1).float()

        #compute U_sum and U_contri using the formulas defined above
        U_sum = self.B - U_unclamped - sparse.ComputeSparseMv(self.Gw, self.GwTr, 1-L)
        if(squeeze==True):
            Ucontr = L[child_ids] - self.invW.unsqueeze(1) * U_sum[par_ids] #size m
        else:
            Ucontr = L[child_ids] - self.invW * U_sum[par_ids] #size m

        # clamp L_contri to range [0,1].
        Ucontr = 1 - clamp(Ucontr, 0, 1)
        # aggregate U contributions from different parents using gen_downward.
        U = gen_downward(self.G, Ucontr, U, self.down_orphans, False)
        return(L, U)


    def logical_loss(self, learn_m, gW, gB, squeeze=True):
        '''
         Routine for computing logical loss for And Net.
            Parameters:
                    learn_m (int): Number of learnable paramters. 
                    gW (Tensor): Tensor of dim gLNN._nnz() specifying weight values
                                 (includes both learnable and non-learnable parameters)
                    gB(Tensor): Tensor of dim gLNN.size()[0] specifying bias parameters
    

            Formula:
                   \sum_v (\sum_{u\in in(v)}{relu(b(v)-w(u,v)})/|in(v)|
                   where for a node v, in(v) is the set of nodes u that have learnable edges to v.
            Steps:
                 1. From G, create a subgraph G_learn that has only the learnable edges.
                    Note that gLNN was constructed such that learnable edges come first in it. 
                    Thus all edges of G with wptr values < learn_m are learnable.

                    Create wptr_learn,  row_learn, col_learn by masking out the condition (wptr < learn_m)
                    on wptr, G._indices()[0], G._indices()[1] respectively.
                   
                    G_learn is created with indices as (row_learn, col_learn) and nnz size of wptr_learn.
                    This is done by calling the canonize routine in construct_model.
        
                  2. Get the weight values W of the edges and bias values B of the nodes in G_learn.
                  3. Set W_learn = relu(B-W). This stores the first term inside the summation of the RHS in the formula.
                  4. Do a row sum on G_learn (unweighted). This will determine |in(v)| for each node v, i.e., fan_in.
                     Similarly, do a row sum on G_learn with weights W_learn to get A1.
                     A1 will determine for each node v the term
                      (\sum_{u\in in(v)}{relu(b(v)-w(u,v)}).
                     Diving these two terms gives the required loss.

                  Note: For leaf nodes of G_learn, both the numerator and denominators obtained in step 4 will be 0.
                  To avoid this boundary case, the denominator is set to 1 for such nodes. Thus fan_in should be replaced by
                  'fan_in + orphan' before doing the division.

        '''
        #create a matrix G_learn with only learnable edges
        G = self.G
        if(learn_m == None): learn_m = G._nnz()
        mask = self.wptr < learn_m
        wptr_learn = torch.masked_select(self.wptr, mask)
        row_learn = torch.masked_select(G._indices()[0], mask)
        col_learn = torch.masked_select(G._indices()[1], mask)
        indices_learn = torch.stack([row_learn, col_learn])
        G_learn = torch.sparse.FloatTensor(indices_learn, torch.ones(wptr_learn.shape[0],device=util.gen_data.gpu_device), G.size())
        G_learn, wptr_learn = construct_model.canonize(G_learn, wptr_learn)

        # Get the weight values W of the edges and bias values B of the nodes in G_learn.
        W = gW[wptr_learn]
        if(squeeze==True):
            B = gB[self.bptr].unsqueeze(1)
        else:
            B = gB[self.bptr]

        # Construct weighted version of G_learn with  W_learn = relu(B-W)
        row_learn = G_learn._indices()[0]
        W_learn = F.relu(B[row_learn]- W)

        # Do a row sum on the weighted G_learn.  This returns for every node v, the term (\sum_{u\in in(v)}{relu(b(v)-w(u,v)})
        if(squeeze==True):
            A1 = sparse.ComputeSparseRowSum.apply(G_learn, W_learn).unsqueeze(1)
        else:
            A1 = sparse.ComputeSparseRowSum.apply(G_learn, W_learn)

        # Do a row sum on the G_learn. This returns |in(v)| for every node v.
        fan_in = torch.sparse.sum(G_learn, 1).to_dense()
  
        # For nodes with zero |in(v)|, replace the fan_in by 1 to avoid division by zero error. 
        if(squeeze==True):
            orphans = (fan_in == 0).unsqueeze(1)
        else:
            orphans = (fan_in == 0)
        fan_in_one = fan_in + orphans

        # determine logical loss.
        logical_loss  = torch.sum(A1/fan_in_one) * util.gen_data.config["logical_mult"]
        return(logical_loss)

class NotNet:
    '''
    Class to define and operate on the Not Net of the graph. 
    Salient class members:
                    G (SparseTensor): Sparse Tensor defining connectivity of nodes in the Not Net.
                    down_orphans (Boolean Tensor): True for nodes which do not have any parent.
                    up_orphans (Boolean Tensor): True for nodes which do not have any child.
    '''
    def __init__(self, G, wptr, bptr, squeeze=True):
        '''
         Routine for initializing the class members.
            Parameters:
                    G (SparseTensor): Sparse Tensor defining connectivity of nodes in the Not Net.
                    wptr (Tensor): Tensor of dimension  G._nnz() specifying the mapping of Not Net edges to gLNN.
                    bptr (Tensor): Tensor of dimension  G.size()[0] specifying the mapping of Not Net nodes to gLNN. 
                    Note: Currently wptr and bptr are not used since all edges of Not Net have unit weight.
    
        '''
        assert(G.is_coalesced()) 
        self.G = G
        self.n = G.size()[0]
        self.nnz = G._nnz()
        #self.wptr = wptr
        #self.bptr = bptr

        fan_out = torch.sparse.sum(self.G, 0).to_dense()
        fan_in = torch.sparse.sum(self.G, 1).to_dense()
        if(squeeze==True):
            self.down_orphans = (fan_out == 0).unsqueeze(1)
            self.up_orphans = (fan_in == 0).unsqueeze(1)
        else:
            self.down_orphans = (fan_out == 0)
            self.up_orphans = (fan_in == 0)


    def to_device(self, device):
        '''
         Routine to move class elements to cpu/gpu.
            Parameters:
                    device : CPU or particular GPU 
    
        '''
        self.G = self.G.to(device)
        self.down_orphans = self.down_orphans.to(device)
        self.up_orphans = self.up_orphans.to(device)

    def upward(self, L, U):
        '''
         Routine for upward inference of Not Net.
            Parameters:
                    L (Tensor): Lower Bound Tensor of nodes.
                    U (Tensor): Upper Bound Tensor of nodes.
   
            Formula:
                    L_up[parent] = (1 - U[child]) 
                    U_up[parent] = (1 - L[child])
                    return smax(L, L_up) and smin(U, U_up)

            Steps:
                    1. Store 1-L and 1-U in temporary buffers Lcomp and Ucomp respectively.
                    2. Apply formula 1 and 2.
                    3. Do the aggregation. No update required for nodes that don't have children. 
        '''
        par_id = self.G._indices()[0]
        child_id = self.G._indices()[1]
        L_up = torch.zeros(self.n,device=util.gen_data.gpu_device)
        U_up = torch.zeros(self.n,device=util.gen_data.gpu_device)

        # Define Lcomp, Ucomp
        Lcomp = 1 - L
        Ucomp = 1 - U

        # Apply upward inference formula
        L_up[par_id] = Ucomp[child_id]
        U_up[par_id] = Lcomp[child_id]

        # Aggregate. No update required for nodes that don't have children.
        Lnew = self.up_orphans * L + (~self.up_orphans) * aggregate(L, L_up, lower=True)
        Unew = self.up_orphans * U + (~self.up_orphans) * aggregate(U, U_up, lower=False)
        return Lnew, Unew

    def downward(self, L, U):
        '''
         Routine for downward inference of Not Net.
            Parameters:
                    L (Tensor): Lower Bound Tensor of nodes.
                    U (Tensor): Upper Bound Tensor of nodes.
   
            Formula:
                    Recall in upward phase,
                    L_up[parent] = (1 - U[child]) 
                    U_up[parent] = (1 - L[child])
                    Therfore, downward contribution to child from a parent is,
                       U_contri[child, parent] = 1 - L[parent]  
                       L_contri[child, parent] = 1 - U[parent]
                    We then need to aggregate contributions from all parents for a child.

            Steps:
                    1. Determine downward contributions from each parent in Lcontr and Ucontr
                       Lcontr and Ucontr are tensors of  dimension G._nnz() such that 
                       Lcontr[i] (Ucontr[i]) = Lower (Upper) bound update contributed by node 
                                               G._indices()[0][i] to G._indices()[1][i]
                    2. Do the aggregate of contributions from different parents using gen_downward.
        '''
        par_ids = self.G._indices()[0]
        Lcontr = 1 - U[par_ids] #size m
        L = gen_downward(self.G, Lcontr, L, self.down_orphans, True)
        Ucontr = 1 - L[par_ids] #size m
        U = gen_downward(self.G, Ucontr, U, self.down_orphans, False)
        return(L, U)
