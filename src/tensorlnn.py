import torch
import torch.distributed as dist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.utils.checkpoint as checkpoint
import sys
import os
import socket

import util
import sparse
import gates


def create_G(nvar):
    nnodes =  nvar + 1
    nedges =  nvar
    row = torch.zeros(nedges, dtype=torch.long)
    column = torch.arange(1, nnodes, dtype=torch.long)
    index = torch.stack((row, column), 0)
    value = torch.ones(nedges)
    G = torch.sparse_coo_tensor(index, value, (nnodes, nnodes)).to(util.gen_data.gpu_device)
    G = G.coalesce()
    wptr = torch.arange(0, nedges, dtype=torch.long)
    bptr = torch.arange(0, nnodes, dtype=torch.long)
    return(G, wptr, bptr)



class NeuralNet(nn.Module):
    def __init__(self, nvar, gpu_device, nepochs, lr, optimizer):
        super(NeuralNet, self).__init__()
        util.setup_values(gpu_device, nepochs, lr, optimizer)
        G, wptr, bptr = create_G(nvar)
        self.and_net = gates.AndNet(G, wptr, bptr)
        self.n = nvar + 1 
        self.m = nvar
        self.learn_m = nvar
        self.gW = None #|gW| = nnz(gLNN)
        self.gB = None #|gB| = gLNN.size
        self.pW = nn.Parameter(torch.ones(self.m))
        self.pB = nn.Parameter(torch.ones(self.n))
        self.to(util.gen_data.gpu_device)
        self.epoch = 0
        self.optimizer = None

    def print(self):
        print('*** W *****')
        print(self.gW)
        print('*** B *****')
        print(self.gB)

    def vacuity_loss(self):
        vacuity_mult = util.gen_data.config["vacuity_mult"]
        ones = torch.ones(self.n, device=util.gen_data.gpu_device)
        vac_loss = vacuity_mult * torch.sum((ones-self.pB)*(ones-self.pB)) 
        return(vac_loss)

    def bound_loss(self, L, U):
        bound_mult = util.gen_data.config["bound_mult"]
        gap_slope = util.gen_data.config["gap_slope"]
        contra_slope = util.gen_data.config["contra_slope"]
        gap_loss = bound_mult * torch.sum((F.relu(U - L) * gap_slope) * (F.relu(U - L) * gap_slope))
        contra_loss = bound_mult * torch.sum((F.relu(L-U) * contra_slope)*(F.relu(L-U) * contra_slope))
        return(gap_loss, contra_loss)

    def supervised_loss(self, L, U):
        L = L[self.gt_lids] #glids x batch_size
        U = U[self.gt_lids]
        loss_fn = nn.MSELoss()
        loss = (loss_fn(L, self.gtL) + loss_fn(U, self.gtU))/2
        return(loss)

    def multi_steps(self, L, U, nitr):
        for j in range(nitr):
            if(util.gen_data.phase_print): util.my_print('itr = %d' % (j))
            L, U = self.and_net.upward(L, U)
        return(L, U)

    def multi_steps_inf(self, L, U):
        while True:
            L0, U0 = L, U
            L, U = self.and_net.upward(L, U)
            #check max change in L, U
            delta = torch.max(torch.cat((torch.abs(L-L0), torch.abs(U-U0)))).item()
            if(delta < util.gen_data.config["eps"]):
                break
        return(L, U)

    def forward(self, infer=False):
        non_param_w = torch.ones(self.m - self.learn_m, device=util.gen_data.gpu_device)
        self.gW = torch.cat( (self.pW, non_param_w), dim=0)
        self.gB = self.pB

        self.and_net.setup_weights(self.gW, self.gB)
        if(infer==True):
            L, U = self.multi_steps_inf(self.L0, self.U0)
        else:
            L, U = self.multi_steps(self.L0, self.U0, util.gen_data.config["inf_steps"])

        assert((torch.max(L) <= 1) and (torch.min(L) >= 0))
        assert((torch.max(U) <= 1) and (torch.min(U) >= 0))
        return(L, U)

    def grad_aggregate(self):
        return
        dist.all_reduce(self.pW.grad.data, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.pB.grad.data, op=dist.ReduceOp.SUM)

    def display_weights(self):
        pW = self.pW.detach().to(util.gen_data.cpu_device)
        pB = self.pB.detach().to(util.gen_data.cpu_device)
        return pB, pW

    def model_store(self, fname, epoch):
        ind = self.and_net.G._indices()
        ind = ind[:, 0:self.learn_m]
        ind = ind.numpy()
        pW = self.pW.detach().to(util.gen_data.cpu_device).numpy()
        pB = self.pB.detach().to(util.gen_data.cpu_device).numpy()
        np.savez_compressed(fname + '.npz', W_ind = ind, W_vals = pW, bias = pB)
        checkpoint = {
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, fname + '_optim')

    def model_load(self, fname):
        gpu_device = util.gen_data.gpu_device
        optimizer = util.gen_data.config["optimizer"]
        lr = util.gen_data.config["lr"]
        data = np.load(fname + '.npz')
        ind = data['W_ind']
        vals = data['W_vals']
        bias = data['bias']
        vals = torch.from_numpy(vals).type(torch.FloatTensor)
        bias = torch.from_numpy(bias).type(torch.FloatTensor)
        self.pW.data = vals.to(gpu_device)
        self.pB.data = bias.to(gpu_device)

        checkpoint = torch.load(fname + '_optim')
        if(optimizer == 'AdamW'): self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if(optimizer == 'SGD'): self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']

    
    def infer(self,  full_X):
        config = util.gen_data.config

        ones = torch.ones(full_X.size(0),1)
        zeroes = torch.zeros(full_X.size(0),1)
        L0 = torch.cat((zeroes, full_X),1)
        U0 = torch.cat((ones, full_X),1)
        self.gt_lids = torch.tensor([0], dtype=torch.long)
        self.L0 = L0.transpose(0, 1)
        self.U0 = U0.transpose(0, 1)

        st = time.time()
        self.L0 = self.L0.to(util.gen_data.gpu_device)
        self.U0 = self.U0.to(util.gen_data.gpu_device)
        self.and_net.to_device(util.gen_data.gpu_device)


        with torch.no_grad():
            L, U = self.forward(infer=True)
        self.L0 = self.L0.to(util.gen_data.cpu_device)
        self.U0 = self.U0.to(util.gen_data.cpu_device)
        self.and_net.to_device(util.gen_data.cpu_device)
        self.gtL = self.gtL.to(util.gen_data.cpu_device)
        self.gtU = self.gtU.to(util.gen_data.cpu_device)
        L = L[self.gt_lids] 
        U = U[self.gt_lids]
        et = time.time()
        L = L.to(util.gen_data.cpu_device)
        U = U.to(util.gen_data.cpu_device)
        return L, U, et -st
    '''
        util.my_print('--> Infer Time  = %.1f' % (et - st))
        util.my_print('--> GPU mem: %d' % (util.GPU_mem()))
        bptr = univ.and_net.bptr.numpy()
        nnz = bptr.shape[0]
        print(bptr.shape)
        nn = self.n
        np.savez_compressed(fname + '.npz', Bound_n = nn, Bound_nnz=nnz, Bound_ind=bptr, LBound_val = L, UBound_val=U)

    '''

    def train(self, full_X, Y):
    #def train(self, L0, U0, gtL, gtU, gt_lids):
        config = util.gen_data.config
        if self.optimizer ==None:
            if(config["optimizer"] == 'AdamW'): self.optimizer = torch.optim.AdamW(self.parameters(), lr=config["lr"])
            if(config["optimizer"] == 'SGD'): self.optimizer = torch.optim.SGD(self.parameters(), lr=config["lr"])

        st_train = time.time()
        
        ones = torch.ones(full_X.size(0),1)
        zeroes = torch.zeros(full_X.size(0),1)
        L0 = torch.cat((zeroes, full_X),1)
        U0 = torch.cat((ones, full_X),1)
        self.gt_lids = torch.tensor([0], dtype=torch.long)

        self.L0 = L0.transpose(0, 1)
        self.U0 = U0.transpose(0, 1)
        self.gtL = Y.transpose(0, 1)
        self.gtU = self.gtL
        ''' 
        self.L0 = L0
        self.U0 = U0
        self.gtL = gtL
        self.gtU = gtU
        self.gt_lids = gt_lids
        '''

        self.L0 = self.L0.to(util.gen_data.gpu_device)
        self.U0 = self.U0.to(util.gen_data.gpu_device)
        self.gtL = self.gtL.to(util.gen_data.gpu_device)
        self.gtU = self.gtU.to(util.gen_data.gpu_device)
        self.gt_lids = self.gt_lids.to(util.gen_data.gpu_device)
        self.and_net.to_device(util.gen_data.gpu_device)


        for ep in range(self.epoch, config["nepochs"]):
            ep_gap_loss, ep_contra_loss, ep_logical_loss = 0, 0, 0

            L, U = self.forward()
            total_loss = self.supervised_loss(L, U)

            total_loss.backward()

            sys.stdout.flush()
            self.grad_aggregate()
            self.optimizer.step()
            self.optimizer.zero_grad()

            with torch.no_grad():
                self.pW.data = self.pW.data.clamp(min=config["w_clamp_thr_lb"], max=config["w_clamp_thr_ub"])
                self.pB.data = self.pB.data.clamp(min=config["bias_clamp_thr_lb"], max=config["bias_clamp_thr_ub"])


            #util.master_print('--> Epoch %d : Supervised = %f' % (ep, total_loss))

        self.L0 = self.L0.to(util.gen_data.cpu_device)
        self.U0 = self.U0.to(util.gen_data.cpu_device)
        self.and_net.to_device(util.gen_data.cpu_device)
        self.gtL = self.gtL.to(util.gen_data.cpu_device)
        self.gtU = self.gtU.to(util.gen_data.cpu_device)
        self.gt_lids = self.gt_lids.to(util.gen_data.cpu_device)


        et_train = time.time()        
        #util.my_print('--> Time per epoch = %.1f' % ((et_train - st_train)/config["nepochs"]))
        #util.my_print('--> Time for all epochs = %.1f' % ((et_train - st_train)))
        #util.my_print('--> GPU mem: %d' % (util.GPU_mem()))

        return total_loss, (et_train - st_train)
