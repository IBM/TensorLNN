import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.utils.checkpoint as checkpoint
import util

def spmm_grad(Gw, Hin, gradHout):
    return(gradW, gradHin)

def MaskedDenseMM(X, Y, rows, cols):
    """Helper routine for Backward computation for spmm
    Z = X[rows,:] x Y[cols,:]
    Args:
        X: dense matrix
        Y: dense matrix
        rows: row indices of X
        cols: row indices of Y

    Returns:
        Z as defined above
    """
    #Z = X * Y
    m = rows.shape[0]
    assert(m == cols.shape[0])
    assert(X.shape == Y.shape)
    xrows = X[rows,:]
    yrows = Y[cols,:]
    Z = torch.sum(xrows * yrows, dim=1)
    return(Z)


class CreateGw(torch.autograd.Function):
    """Class to define forward and backward
    for setting up a sparse tensor.
    """
    @staticmethod
    def forward(ctx, G, W):
        assert(G.is_coalesced()) 
        Gw = torch.sparse.FloatTensor(G._indices(), W, G.size())
        Gw = Gw.coalesce()   #This declaration
        return(Gw)

    @staticmethod
    def backward(ctx, gradGw):
        gradGw = gradGw.coalesce()    #Unavoidable
        gradW = gradGw._values()
        return(None, gradW)


class ComputeSparseMM(torch.autograd.Function):
    """Class to define forward and backward
    for sparese mm.
    """
    # Hout = Gw Hin
    @staticmethod
    def forward(ctx, Gw, GwTr, Hin):
        assert(Gw.is_coalesced())
        if(GwTr != None): assert(GwTr.is_coalesced())
        Hout = torch.sparse.mm(Gw, Hin)   #Fast spmm
        ctx.save_for_backward(Gw, GwTr, Hin)
        return(Hout)

    @staticmethod
    def backward(ctx, gradHout):
        Gw, GwTr, Hin = ctx.saved_tensors
        if(GwTr == None): GwTr = Gw.transpose(0, 1).coalesce()
        gradHin = torch.sparse.mm(GwTr, gradHout) #Fast spmm
        rows = Gw._indices()[0]
        cols = Gw._indices()[1]
        gradW = MaskedDenseMM(gradHout, Hin, rows, cols)
        gradGw = torch.sparse.FloatTensor(Gw._indices(), gradW, Gw.size()).coalesce() #Declarative
        return(gradGw, None, gradHin)


def ComputeSparseMv(Gw, GwTr, v):
    if(v.ndim==1):
        v = torch.unsqueeze(v, 1)
        ans = ComputeSparseMM.apply(Gw, GwTr, v)
        return torch.squeeze(ans)
    else:
        return ComputeSparseMM.apply(Gw, GwTr, v)



def matrix_vector_fp(A, v, mult_precision='float', add_precision = 'float'):
    assert(A.is_coalesced())

    if mult_precision=='half':
        v_fp = v.half()
        A_val = A._values().half()
    else:
        v_fp = v
        A_val = A._values()
    v_col = v_fp[A._indices()[1]]
    prod = v_col * A_val

    if add_precision == 'half':
        B = torch.sparse.HalfTensor(torch.stack([A._indices()[0],A._indices()[0]]), prod.half(), A.size())
    else:
        B = torch.sparse.FloatTensor(torch.stack([A._indices()[0],A._indices()[0]]), prod.float(), A.size())

    B=B.coalesce()
    result = torch.zeros(A.size()[0],device=util.gen_data.gpu_device)
    result[B._indices()[0]] = B._values().float()

    return result



class ComputeSparseMv_fp(torch.autograd.Function):
    """Class to define forward and backward
    for sparese mm.
    """
    # Hout = Gw Hin
    @staticmethod
    def forward(ctx, Gw, GwTr, Hin):
        assert(Gw.is_coalesced())
        if(GwTr != None): assert(GwTr.is_coalesced())
        Hout = matrix_vector_fp(Gw, Hin)   #Fast spmm
        ctx.save_for_backward(Gw, GwTr, Hin)
        return(Hout)

    @staticmethod
    def backward(ctx, gradHout):
        Gw, GwTr, Hin = ctx.saved_tensors
        if(GwTr == None): GwTr = Gw.transpose(0, 1).coalesce()
        gradHin = matrix_vector_fp(GwTr, gradHout) #Fast spmm
        rows = Gw._indices()[0]
        cols = Gw._indices()[1]
        gradHout_extend = torch.unsqueeze(gradHout, 1)
        Hin_extend = torch.unsqueeze(Hin, 1)
        gradW = MaskedDenseMM(gradHout_extend, Hin_extend, rows, cols)
        gradGw = torch.sparse.FloatTensor(Gw._indices(), gradW, Gw.size()).coalesce() #Declarative
        return(gradGw, None, gradHin)


class ComputeSparseMvNew(torch.autograd.Function):
    """Class to define forward and backward
    for sparese mm.
    """
    # Hout = Gw Hin
    @staticmethod
    def forward(ctx, G, W, Hin):
        assert(G.is_coalesced())
        Gw = torch.sparse.FloatTensor(G._indices(), W, G.size())
        Hout = torch.sparse.mm(Gw, torch.unsqueeze(Hin, 1)).squeeze()   #Fast spmm
        ctx.save_for_backward(G, W, Hin)
        return(Hout)

    @staticmethod
    def backward(ctx, gradHout):
        G, W, Hin = ctx.saved_tensors
        Gw = torch.sparse.FloatTensor(G._indices(), W, G.size())
        GwTr = Gw.transpose(0, 1)
        gradHin = torch.sparse.mm(GwTr, torch.unsqueeze(gradHout,1)).squeeze()#Fast spmm
        rows = Gw._indices()[0]
        cols = Gw._indices()[1]
        gradW = MaskedDenseMM(torch.unsqueeze(gradHout,1), torch.unsqueeze(Hin, 1), rows, cols)
        return(None, gradW, gradHin)

class ComputeSparseColSum(torch.autograd.Function):
    """Class to define forward and backward
    for sparese mm.
    """
    # Hout = Gw Hin
    @staticmethod
    def forward(ctx, G, vals):
        assert(G.is_coalesced())
        assert(G._nnz() == vals.size()[0])
        if(vals.ndim==1):
            Gvals = torch.sparse.FloatTensor(G._indices(), vals, G.size())
        else:
            N = G.size()[0]
            K = vals.size()[1]
            Gvals = torch.sparse.FloatTensor(G._indices(), vals, torch.Size([N, N, K]))
        colSum = torch.sparse.sum(Gvals,dim=0).to_dense()
        ctx.save_for_backward(G)
        return(colSum)

    @staticmethod
    def backward(ctx, gradColsum):
        G = ctx.saved_tensors[0]
        rows = G._indices()[0]
        cols = G._indices()[1]
        gradVals = gradColsum[cols]
        return(None, gradVals)

class ComputeSparseRowSum(torch.autograd.Function):
    """Class to define forward and backward
    for sparese mm.
    """
    # Hout = Gw Hin
    @staticmethod
    def forward(ctx, G, vals):
        assert(G.is_coalesced())
        assert(G._nnz() == vals.size()[0])
        Gvals = torch.sparse.FloatTensor(G._indices(), vals, G.size())
        rowSum = torch.sparse.sum(Gvals,dim=1).to_dense()
        ctx.save_for_backward(G)
        return(rowSum)

    @staticmethod
    def backward(ctx, gradRowsum):
        G = ctx.saved_tensors[0]
        rows = G._indices()[0]
        cols = G._indices()[1]
        gradVals = gradRowsum[rows]
        return(None, gradVals)

