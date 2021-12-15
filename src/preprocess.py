import numpy as np
import torch
import time
import sys
import os
import scipy
import json
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import util

def read_LNN_csr(adjfile):
    '''
    Routine to read graph from npz file.
            Parameter:
                    adjfile: npz file with fields data, indices, indptr, shape.
    Steps : 
         1. Create csr_matrix LNN_csr of shape=shape with arguements of data, indices, indptr. 
         2. Create FloatTensor LNN_tensor from LNN_csr.
            Returns:
                    LNN_tensor : Sparse Tensor generated using above steps.
    '''
    Gw = np.load(adjfile)
    LNN_csr = csr_matrix((Gw["data"], Gw["indices"], Gw["indptr"]), shape=Gw["shape"])
    LNN_coo = LNN_csr.tocoo()
    nnz = len(LNN_coo.row.tolist())
    LNN_tensor = torch.sparse.FloatTensor(torch.LongTensor([LNN_coo.row.tolist(), LNN_coo.col.tolist()]), torch.ones(nnz), torch.Size(list(LNN_coo.shape)))
    LNN_tensor = LNN_tensor.to(util.gen_data.gpu_device)
    return LNN_tensor

def construct_gLNN():
    '''
    Routine to construct the global LNN.
    Steps : 
         1. Read the global npz files to get G (entire graph) and learnG (learnable subgraph).
         2. Rearranges elements in G so that learnable edges come first to form gLNN.

            Returns:
                    gLNN : Sparse Tensor generated using above steps.
                    learn_m : Number of learnable edges.
    '''
    univ_dir = os.path.join(util.gen_data.config['univ_dir'], 'global')
    gLNN_path = os.path.join(univ_dir, 'global_adj.npz')
    gLearn_path = os.path.join(univ_dir, 'global_Learnable_edges_adj.npz')
    G = read_LNN_csr(gLNN_path).coalesce()
    learnG = read_LNN_csr(gLearn_path)

    # G : global LNN
    # learnG : submatrix of G with only learnable edges
    # Rearranges elements in G so that learnable edges come first.
    # This is done by creating a sparseTensor comb_G of same nnz pattern as G 
    # having values 2 for the learnG indices and 1 for the rest.

    comb_ind = torch.cat((G._indices(), learnG._indices()), dim=1)
    vals = torch.ones(G._nnz() + learnG._nnz(), device=util.gen_data.gpu_device)
    comb_G = torch.sparse.FloatTensor(comb_ind, vals, G.size())
    comb_G = comb_G.coalesce()
    assert(comb_G._nnz() == G._nnz())

    # Determine the index positions  with values 2 (1): learn_pos (nonlearn_pos)
    # Create sparseTensor gLNN with learn_pos positioned indices of G followed by
    # nonlearn_pos positioned indices of G.

    learn_pos = torch.nonzero(comb_G._values() == 2).squeeze()
    nonlearn_pos = torch.nonzero(comb_G._values() == 1).squeeze()
    old_row = comb_G._indices()[0]
    new_row = torch.cat((old_row[learn_pos], old_row[nonlearn_pos]), dim=0)
    old_col = comb_G._indices()[1]
    new_col = torch.cat((old_col[learn_pos], old_col[nonlearn_pos]), dim=0)
    new_ind = torch.stack((new_row, new_col), 0)
    new_vals = torch.ones(comb_G._nnz(), device=util.gen_data.gpu_device)
    gLNN = torch.sparse.FloatTensor(new_ind, new_vals, G.size())
    learn_m = learnG._nnz()
    return(gLNN, learn_m)


def construct_single_univ(gLNN, univ_id):
    '''
    Routine to determine data structures for the ANDNet, NOTnet and bounds of the universe.
            Parameters:
                    gLNN : Global LNN
                    univ_id (int): Universe id
            Returns:
                    lAndG : Sparse Tensor for AndNet of univ_id universe with nodes and edges numbered locally 
                    and_wptr : Mapping from local edge indices to global edge indices for lAndG
                    and_bptr : Mapping from local node indices to global node indices for lAndG and lNotG
                    lNotG : Sparse Tensor for NotNet of univ_id universe with nodes and edges numbered locally
                    not_wptr : Mapping from local edge indices to global edge indices for lNotG
                    not_bptr : Mapping from local node indices to global node indices for lAndG and lNotG
                    lBounds : Nodes (numbered locally) whose bounds should be initialized
                    

    Steps :
         1. Read the npz files for AndNet and NotNet of the universe to get gAndG and gNotG respectively.
         2. Note gAndG and gNotG sparse tensors have nodes and edges given by global numbering, whereas to operate on
            them we need local numbering. Hence we need to construct lAndG and lNotG from gAndG and gNotG respectively,
            The intuitive idea of the indices mapping is explained below.
            Suppose gLNN only has 10 nodes numbered 0 to 9 and 
            gAndG._indices() = ([3, 5, 8], [7, 3, 6]),  gNotG._indices() = ([2, 6], [7, 8]). 
            Thus gAndG and gNotG involves only a subgraph with node set {2, 3, 5, 6, 7, 8}.
            Thus and_bptr = not_bptr = [2, 3, 5, 6, 7, 8]; lAndG._indices() = ([1, 2, 5],[4, 1, 3]), lNotG._indices() = ([0, 3], [4, 5]).
            Further suppose the edges [2, 7], [3, 7], [5, 3], [6, 8] and [8, 6] refer to non-zero indices 
            i1, i2, i3, i4, i5 in gLNN. Then and_wptr = [i2, i3, i5], and not_wptr = [i1, i4].             
          3. Read bounds file. Get the global indexed nodes which should be initialized. Convert them to local numbering.
    '''
    gN = gLNN.size()[0]
    # Read the npz files for AndNet and NotNet of the universe to get gAndG and gNotG respectively 
    univ_dir = os.path.join(util.gen_data.config['univ_dir'], 'local', str(univ_id))
    and_adjfile = os.path.join(univ_dir, 'And_adj_3e+05.npz')
    not_adjfile = os.path.join(univ_dir, 'Not_adj_3e+05.npz')
    gAndG = read_LNN_csr(and_adjfile).coalesce()
    gNotG = read_LNN_csr(not_adjfile).coalesce()

    # determine set of nodes combined in gAndG and gNotG and store it in gX. 
    comb_ind = torch.cat((gAndG._indices()[0], gAndG._indices()[1], gNotG._indices()[0], gNotG._indices()[1]), dim=0)
    gX =  torch.unique(comb_ind, sorted=True, return_inverse=False) #sorted
    loc_n = gX.numel()

    # gX shall define and_bptr and not_bptr. 
    # To get local indices for lAndG and lNotG, define a mapping g2l_map from ith element of gX to i (for 0<=i<loc_n).
    gX_loc = torch.arange(loc_n, device=util.gen_data.gpu_device)
    g2l_map = torch.sparse.FloatTensor(gX.unsqueeze(0), gX_loc).to_dense()

    # define lAndG with indices mapped locally using g2l_map on gAndG. 
    # all non-zero entries of lAndG set to 1. Size of lAndG = loc_n x loc_n 
    lAndG_row = g2l_map[gAndG._indices()[0]]
    lAndG_col = g2l_map[gAndG._indices()[1]]
    lAndG_ind = torch.stack([lAndG_row, lAndG_col])
    lAndG = torch.sparse.FloatTensor(lAndG_ind, torch.ones(gAndG._nnz(), device=util.gen_data.gpu_device), size=(loc_n, loc_n))
    and_bptr = gX.long()

    # and_wptr is obtained as follows.  A sparseTensor comb_G of same nnz pattern as gLNN is created by
    # (i) comb._G_indices() set by concatenating indices of gLNN and gAndG (note gAndG is a subgraph of gLNN)
    # (ii) comb_G._values() set by concatenating a tensor of index positions in gLNN 
    # with a tensor having all values set to a big number greater than gLNN._nnz().
    # Thus after coalescing comb_G, values of comb_G at any index  absent in gAndG 
    # will be set to the index value itself, while values of comb_G at an index present 
    # in gAndG will be set to (index value + the big number).
    # Getting all values greater than the big number, and subtracting big number from those values will
    # return the GLNN indices of the edges in present in gAndG which is basically and_wptr.
    wptr = torch.arange(gLNN._nnz(), device=util.gen_data.gpu_device)
    comb_ind = torch.cat((gLNN._indices(), gAndG._indices()), dim=1)
    big_num = gLNN._nnz() * 2
    big_val = torch.ones(gAndG._nnz(), device=util.gen_data.gpu_device).long()
    big_val = big_val * big_num
    comb_val = torch.cat((wptr, big_val), dim=0)
    comb_G = torch.sparse.LongTensor(comb_ind, comb_val, size=(gN, gN))
    comb_G = comb_G.coalesce()
    loc_pos = torch.nonzero(comb_G._values() >= big_num).squeeze()
    and_wptr = comb_G._values()[loc_pos] - big_num
    
    # define lNotG with indices mapped locally using g2l_map on gNotG. 
    # all non-zero entries of lNotG set to 1. Size of lNotG = loc_n x loc_n 
    lNotG_row = g2l_map[gNotG._indices()[0]]
    lNotG_col = g2l_map[gNotG._indices()[1]]
    lNotG_ind = torch.stack([lNotG_row, lNotG_col])
    lNotG = torch.sparse.FloatTensor(lNotG_ind, torch.ones(gNotG._nnz(), device=util.gen_data.gpu_device), size=(loc_n, loc_n))
    not_bptr = gX.long()

    # not_wptr is obtained as follows.  A sparseTensor comb_G of same nnz pattern as gLNN is created by
    # (i) comb._G_indices() set by concatenating indices of gLNN and gNotG (note gNotG is a subgraph of gLNN)
    # (ii) comb_G._values() set by concatenating a tensor of index positions in gLNN 
    # with a tensor having all values set to a big number greater than gLNN._nnz().
    # Thus after coalescing comb_G, values of comb_G at any index  absent in gNotG 
    # will be set to the index value itself, while values of comb_G at an index present 
    # in gNotG will be set to (index value + the big number).
    # Getting all values greater than the big number, and subtracting big number from those values will
    # return the GLNN indices of the edges in present in gNotG which is basically not_wptr.
    wptr = torch.arange(gLNN._nnz(), device=util.gen_data.gpu_device)
    comb_ind = torch.cat((gLNN._indices(), gNotG._indices()), dim=1)
    big_num = gLNN._nnz() * 2
    big_val = torch.ones(gNotG._nnz(), device=util.gen_data.gpu_device).long()
    big_val = big_val * big_num
    comb_val = torch.cat((wptr, big_val), dim=0)
    comb_G = torch.sparse.LongTensor(comb_ind, comb_val, size=(gN, gN))
    comb_G = comb_G.coalesce()
    loc_pos = torch.nonzero(comb_G._values() >= big_num).squeeze()
    not_wptr = comb_G._values()[loc_pos] - big_num

    # Read bounds file. Get the global indexed nodes which should be initialized. Convert them to local numbering.
    bounds_file = os.path.join(univ_dir, 'bounds.txt')
    f = open(bounds_file, 'r')
    gBounds = []
    for x in f: gBounds.append(int(x))
    gBounds = torch.tensor(gBounds).long()
    lBounds = g2l_map[gBounds]
    return(lAndG, and_wptr, and_bptr, lNotG, not_wptr, not_bptr, lBounds)

