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
import gates

class Univ:
    '''
    Class for one particular (mega) universe.
    Salient class members:
                    and_net: Object of AndNet class.
                    not_net: Object of NotNet class.
                    L0, U0: Initial bounds on the nodes of the (mega) universe.
                    gW: global weight tensor, i.e., weight of gLNN
                    gB: global bias tensor, i.e., bias of gLNN
    '''
    def __init__(self, and_net, not_net, L0, U0):
        '''
         Routine for initializing the class members.
            Parameters:
                    and_net: Object of AndNet class.
                    not_net: Object of NotNet class.
                    L0, U0: Initial bounds on the nodes of the (mega) universe.
    
        '''
        assert(and_net.n == not_net.n)
        self.n = and_net.n
        self.and_net = and_net
        self.not_net = not_net
        assert(L0.nelement() == self.n)
        assert(U0.nelement() == self.n)
        self.L0 = L0 
        self.U0 = U0
        self.gW = None   #Global W = size of gLNN
        self.gB = None   

    def to_device(self, device):
        '''
         Routine to move class elements to cpu/gpu.
            Parameters:
                    device : CPU or particular GPU 
    
        '''
        self.L0 = self.L0.to(device)
        self.U0 = self.U0.to(device)
        self.and_net.to_device(device)
        self.not_net.to_device(device)
        self.gW = None
        self.gB = None

    def print_stats(self):
        '''
         Routine to print and net and not net dimensions.
    
        '''
        util.my_print('Univ: n = %d and_nnz %d not_nnz %d' % (self.n, self.and_net.nnz, self.not_net.nnz))

    def multi_steps(self, L, U, nitr):
        '''
         Routine defining inference procedure for one epoch.
         Parameters:
                    L, U : Tensors of size self.n Lower and Upper bounds  
                    nitr (int): Number of steps in each epoch/checkpoint segment. 

         Steps:
               1. Setup the weights of the And Net of the (mega) universe from global weights and bias
               2. For nitr steps, repeat 'not upward', 'and upward', 'and downward', 'not downward' in sequence.
        '''
        self.and_net.setup_weights(self.gW, self.gB, squeeze=False)
        for j in range(nitr):
            if(util.gen_data.phase_print): util.my_print('itr = %d' % (j))
            L, U = self.not_net.upward(L, U)
            L, U = self.and_net.upward(L, U, clamp=gates.val_clamp_grad)
            L, U = self.and_net.downward(L, U, squeeze=False, clamp=gates.val_clamp_grad)
            L, U = self.not_net.downward(L, U)
        return(L, U)

    def multi_steps_inf(self, L, U):
        '''
         Routine defining inference procedure for one epoch.
         Parameters:
                    L, U : Tensors of size self.n Lower and Upper bounds  

         Steps:
               1. Setup the weights of the And Net of the (mega) universe from global weights and bias
               2. Repeat 'not upward', 'and upward', 'and downward', 'not downward' in sequence 
                      till the max change in L,U values over two successive step is less than predefined threshold epsilon.
        '''

        self.and_net.setup_weights(self.gW, self.gB, squeeze=False)
        while True:
            if(util.gen_data.phase_print): util.my_print('itr = %d' % (j))
            L0, U0 = L, U
            L, U = self.not_net.upward(L, U)
            L, U = self.and_net.upward(L, U)
            L, U = self.and_net.downward(L, U, squeeze=False)
            L, U = self.not_net.downward(L, U)
            #check max change in L, U
            delta = torch.max(torch.cat((torch.abs(L-L0), torch.abs(U-U0)))).item()
            if(delta < util.gen_data.config["eps"]):
                break
        return(L, U)

    def execute(self, gW, gB, infer=False) :
        '''
         Routine for executing a particular (mega) universe (baseline or checkpoint).
         Parameters:
                    gW: global weight tensor, i.e., weight of gLNN
                    gB: global bias tensor, i.e., bias of gLNN
                    infer (bool): flag for inference (True) or training (False)

         Steps:
               1. Copy the input parameters gW and gB to corresponding class members.
               2. Call appropriate inference procedure based on infer/train and baseline/checkpoint.
        '''
        self.gW = gW
        self.gB = gB
        if(infer==True):
            L, U = self.multi_steps_inf(self.L0, self.U0)
        else:
            if(util.gen_data.config["fwd_method"] == 'baseline'):
                L, U = self.multi_steps(self.L0, self.U0, util.gen_data.config["inf_steps"])
            else:
                L, U = self.chkpt_execute()
        return(L, U)

    def chkpt_execute(self):
        '''
         Wrapper routine for executing a checkpoint segment for training of a particular (mega) universe.
         Steps:
               1. Read the number of steps and checkpoint phases (segments). phases should divide steps.
               2. Define dummy L, U required for pytorch checkpoint procedure. 
               3. Keep track of the direction (forward or backward) and phase number.
               4. For every phase, call the  checkpoint routine chkpt_one_phase.
        '''
        # Read the number of steps and checkpoint phases (segments). phases should divide steps.
        config = util.gen_data.config
        nsteps = config["inf_steps"]
        nphases = config["checkpoint_phases"]
        assert((nsteps % nphases) == 0)
        self.nitr = int(nsteps / nphases)


        # Define  L, U required for pytorch checkpoint procedure.
        # This defining is required to avoid error while setting L and U in actual checkpoint routine
        L = torch.zeros(1,1, requires_grad=True, device=util.gen_data.gpu_device)  #dummy
        U = torch.zeros(1,1, requires_grad=True, device=util.gen_data.gpu_device)  #dummy


        # Keep track of the direction (forward or backward) in util.gen_data.dir, and phase number in self.b
        self.b = 0
        util.gen_data.dir =  'fwd'  #currently set to forward.

        # Loop over nphases
        for b in range(nphases):
            if(util.gen_data.phase_print): util.my_print('-> phase number ', b)
            # call the checkpoint routine chkpt_one_phase
            L, U = checkpoint.checkpoint(self.chkpt_one_phase, L, U)

        # forward done, set b to 'nphases - 1' and util.gen_data.dir to backward.
        self.b = nphases - 1
        util.gen_data.dir =   'bak'

        return(L, U)

    def chkpt_one_phase(self, *inputs):
        '''
         Routine for one phase of checkpoint.

         Steps:
               1. Initialize L, U.
               2. Increment or Decrement phase number depending on forward or backward direction respectively.
               3. Call routine multi_steps 
        '''
        if(self.b == 0): 
            L = self.L0
            U = self.U0
        else: 
            L = inputs[0]
            U = inputs[1]
        if(util.gen_data.dir == 'fwd'): self.b = self.b+1
        else: self.b = self.b - 1

        L, U = self.multi_steps(L, U, self.nitr)
        return(L, U)

    def bound_loss(self, L, U):
        '''
         Routine to compute gap loss and contradiction loss of the (mega) universe given bounds L and U.
         Parameters:
                    L, U : Tensors of size self.n Lower and Upper bounds  

         Formula: 
               Gap Loss = \sum_{v} [ relu(U(v)-L(v)) ]^2 * \beta * \gamma^2  
               Contradiction Loss = \sum_{v} [ relu(L(v)-U(v)) ]^2 * \beta  \zeta^2

         where \beta is the bound_mult, \gamma is gap_slope and \zeta is  contra_slope.
        '''
        bound_mult = util.gen_data.config["bound_mult"]
        gap_slope = util.gen_data.config["gap_slope"]
        contra_slope = util.gen_data.config["contra_slope"]
        gap_loss = bound_mult * torch.sum((F.relu(U - L) * gap_slope) * (F.relu(U - L) * gap_slope))
        contra_loss = bound_mult * torch.sum((F.relu(L-U) * contra_slope)*(F.relu(L-U) * contra_slope))
        return(gap_loss, contra_loss)

class WordNet(nn.Module):
    '''
    Top level class for wordnet sense disambiguation. Involves training/inference over multiple (mega) universes.
    Salient class members:
                    gLNN: SparseTensor for global LNN.
                    learn_m: Number of learnable edges in global LNN.
                    univ_list: List of (Mega) universes to be processed by current gpu/process.
                    gW: global weight tensor, i.e., weight of gLNN
                    gB: global bias tensor, i.e., bias for nodes of gLNN
                    pW: weight parameter tensor, i.e., learnable weights of gLNN
                    pB: bias parameter tensor, i.e., learnable bias for nodes of gLNN
                    epoch: current epoch number, applicable if loading an already partially trained model
                    optimizer: optimizer used for training
                    my_num_univ: number of mega universes to be processed by current process/gpu 
                    g_num_univ: total number of mega universes

                
    '''
    def __init__(self, gLNN, learn_nnz, univ_list):
        '''
         Routine for initializing the class members.
            Parameters:
                    gLNN: global LNN
                    learn_nnz: Number of learnable edges in global LNN.
                    univ_list: list of mega universes to be processed by current process/gpu 
    
        '''
        super(WordNet, self).__init__()
        assert(len(univ_list) != 0)
        self.gLNN = gLNN
        self.n = gLNN.size()[0]
        self.m = gLNN._nnz()
        self.learn_m = learn_nnz
        self.univ_list = univ_list
        self.gW = None #|gW| = nnz(gLNN)
        self.gB = None #|gB| = gLNN.size

        #define learnable parameters pW for learnable weights of gLNN and pB for learnable bias of gLNN
        self.pW = nn.Parameter(torch.ones(self.learn_m))
        self.pB = nn.Parameter(torch.ones(self.n))
        self.to(util.gen_data.gpu_device)

        self.epoch = 0
        self.optimizer = None

        # initialize number of mega universes to be processed by current process/gpu 
        self.my_num_univ = len(univ_list)

        # aggregating across processes/gpus gives total count of mega universes
        self.g_num_univ = int(util.float_agg(self.my_num_univ))

    def print(self):
        ''' routine to print gW and gB'''
        print('*** gW *****')
        print(self.gW)
        print('*** gB *****')
        print(self.gB)

    def vacuity_loss(self):
        '''
         Routine to compute vacuity loss.

         Formula: 
               Vacuity Loss = \sum_{v} \left [1-pB(v)\right]^2 * \nu 

         where \nu is the vacuity_mult. 
        '''
        vacuity_mult = util.gen_data.config["vacuity_mult"]
        ones = torch.ones(self.n, device=util.gen_data.gpu_device)
        vac_loss = vacuity_mult * torch.sum((ones-self.pB)*(ones-self.pB)) 
        return(vac_loss)

    def forward(self, univ, infer=False):
        '''
         Routine for doing forward pass on a particular (mega) universe.
         Parameters:
                    univ: (mega) universe object.
                    infer (bool): flag for inference (True) or training (False)

         Steps:
               1. Create gW from pW by concatenation of non learnable edges.
               2. Call execute routine of (mega) universe univ.
        '''
        non_param_w = torch.ones(self.m - self.learn_m, device=util.gen_data.gpu_device)
        self.gW = torch.cat( (self.pW, non_param_w), dim=0)
        self.gB = self.pB
        L, U  = univ.execute(self.gW, self.gB, infer=infer)
        assert((torch.max(L) <= 1) and (torch.min(L) >= 0))
        assert((torch.max(U) <= 1) and (torch.min(U) >= 0))
        return(L, U)

    def grad_aggregate(self):
        '''
         Routine for pW gradient and pB gradient accumulation across gpus/processes.
        '''
        dist.all_reduce(self.pW.grad.data, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.pB.grad.data, op=dist.ReduceOp.SUM)

    def model_store(self, fname, epoch):
        '''
         Routine for storing trained model weights and model state.
         Parameters:
                    fname: file name to store model. model weights will be stored at <fname>.npz while 
                           model state dictionary will be stored at <fname>_optim
                    epoch (int): epoch till which the model has been trained.

        '''
        # get non zero indices of learnable parameters
        ind = self.gLNN._indices()
        ind = ind[:, 0:self.learn_m]
        ind = ind.numpy()

        # get learnable parameters
        pW = self.pW.detach().to(util.gen_data.cpu_device).numpy()
        pB = self.pB.detach().to(util.gen_data.cpu_device).numpy()

        # save in compressed format
        np.savez_compressed(fname + '.npz', W_ind = ind, W_vals = pW, bias = pB)

        # save state dictionary
        checkpoint = {
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, fname + '_optim')

    def model_load(self, fname):
        '''
         Routine to load a partially trained model (to resume training from that point).
         Parameters:
                    fname: file name to read the model from. model weights will be read from <fname>.npz while 
                           model state dictionary will be read from <fname>_optim

        '''
        # initialize training parameters
        gpu_device = util.gen_data.gpu_device
        optimizer = util.gen_data.config["optimizer"]
        lr = util.gen_data.config["lr"]

        # read npz file to get learnable weights
        data = np.load(fname + '.npz')
        ind = data['W_ind']
        vals = data['W_vals']
        bias = data['bias']
        vals = torch.from_numpy(vals).type(torch.FloatTensor)
        bias = torch.from_numpy(bias).type(torch.FloatTensor)
        self.pW.data = vals.to(gpu_device)
        self.pB.data = bias.to(gpu_device)

        # read model state to define optimizer and epoch
        checkpoint = torch.load(fname + '_optim')
        if(optimizer == 'AdamW'): self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        if(optimizer == 'SGD'): self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']

    def infer(self, univ_id, fname='bounds'):
        '''
         Routine to do inference for a particular (mega) universe.
         Parameters:
                    univ_id: (mega) universe id.
                    fname (string): bounds obtained after inference will be stored in <fname>.npz

        '''
       
        config = util.gen_data.config
        assert((config["fwd_method"] == 'baseline') or (config["fwd_method"] == 'checkpoint'))
        assert(univ_id < num_univ)

        # do a barrier across processes/gpus
        st = util.barr_time()
        ep_st = time.time()

        # get universe object and move the object to gpu
        univ = self.univ_list[univ_id]
        univ.to_device(util.gen_data.gpu_device)

        # call inference routine
        with torch.no_grad():
                L, U = self.forward(univ,infer=True)
                # gap_loss, contra_loss = univ.bound_loss(L, U)
                # logical_loss = univ.and_net.logical_loss(self.learn_m, self.gW, self.gB)
                # vac_loss = self.vacuity_loss()
                # total_loss = gap_loss + contra_loss + logical_loss + vac_loss

        # move universe object back to cpu
        univ.to_device(util.gen_data.cpu_device)
        sys.stdout.flush()

        # do a barrier across processes/gpus
        et = util.barr_time()
        util.my_print('--> Infer Time  = %.1f' % (et - st))
        util.my_print('--> GPU mem: %d' % (util.GPU_mem()))

        # get the mapping of And Net nodes to gLNN 
        bptr = univ.and_net.bptr.numpy()
        nnz = bptr.shape[0]
        print(bptr.shape)
        nn = self.n

        # store bptr, L and U.
        L = L.to(util.gen_data.cpu_device).numpy()
        U = U.to(util.gen_data.cpu_device).numpy()
        np.savez_compressed(fname + '.npz', Bound_n = nn, Bound_nnz=nnz, Bound_ind=bptr, LBound_val = L, UBound_val=U)

    def train(self):
        '''
         Routine to do training for WordNet Sense Disambiguation.


        '''
        # set up optimizer if not set already
        config = util.gen_data.config
        assert((config["fwd_method"] == 'baseline') or (config["fwd_method"] == 'checkpoint'))
        if self.optimizer ==None:
            if(config["optimizer"] == 'AdamW'): self.optimizer = torch.optim.AdamW(self.parameters(), lr=config["lr"])
            if(config["optimizer"] == 'SGD'): self.optimizer = torch.optim.SGD(self.parameters(), lr=config["lr"])

        st_train = time.time()
        # loop over epochs
        for ep in range(self.epoch, config["nepochs"]):
            ep_gap_loss, ep_contra_loss, ep_logical_loss = 0, 0, 0
            # loop over (mega) universes
            for univ_id in range(self.my_num_univ):
                util.master_print('Epoch %d univ %d/%d' % (ep, univ_id, self.my_num_univ))

                # get univ object and move it to gpu
                univ = self.univ_list[univ_id]
                univ.to_device(util.gen_data.gpu_device)

                # do a forward inference for the (mega) universe
                L, U = self.forward(univ)

                # determine loss given the (mega) universe and bounds 
                gap_loss, contra_loss = univ.bound_loss(L, U)
                logical_loss = univ.and_net.logical_loss(self.learn_m, self.gW, self.gB, squeeze=False)
                vac_loss = self.vacuity_loss()/self.g_num_univ
                total_loss = gap_loss + contra_loss + logical_loss + vac_loss
                # propagate loss backwards 
                total_loss.backward()

                # loss is additive on (mega) universes. Hence we can just do a gradient aggregate.
                # compute cumulative loss across multiple (mega) universes processed by this gpu
                ep_gap_loss += gap_loss.item()   # only for print purposes
                ep_contra_loss += contra_loss.item()   # only for print purposes
                ep_logical_loss += logical_loss.item()  # only for print purposes

                # move the universe back to cpu
                univ.to_device(util.gen_data.cpu_device)

            sys.stdout.flush()

            #  gradient aggregate across gpus/processes working on different (mega) universes
            #  we can do this since the loss is additive on (mega) universes.
            self.grad_aggregate()

            # update learnable parameters and reset gradients
            self.optimizer.step()
            self.optimizer.zero_grad()

            # clamp learnable parameters in threshold range given in config json
            self.pW.data = self.pW.data.clamp(min=config["clamp_thr_lb"], max=config["clamp_thr_ub"])
            self.pB.data = self.pB.data.clamp(min=config["clamp_thr_lb"], max=config["clamp_thr_ub"])

            # aggregate cumulative loss across all gpus. 
            # this will produce loss across all (mega) universes. 
            ep_gap_loss = util.float_agg(ep_gap_loss)
            ep_contra_loss = util.float_agg(ep_contra_loss)
            ep_logical_loss= util.float_agg(ep_logical_loss)

            util.master_print('--> Epoch %d : Logical =  %f Vacuity =  %f Gap = %f Contra = %f' % (ep, ep_logical_loss, self.vacuity_loss(), ep_gap_loss, ep_contra_loss))

        et_train = time.time()        
        util.my_print('--> Time per epoch = %.1f' % ((et_train - st_train)/config["nepochs"]))
        util.my_print('--> GPU mem: %d' % (util.GPU_mem()))

