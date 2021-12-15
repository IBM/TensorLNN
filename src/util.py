import torch
import torch.distributed as dist
import torch
import time
import sys
import os
import socket
import numpy as np

def float_agg(my_val):
    '''
    Returns the aggregate of values across all processes/gpus.

            Parameters:
                    my_val (float): values of individual processes/gpus.

            Returns:
                    Aggregate of my_val across all processes/gpus.
    '''
    my_val = torch.tensor([my_val]).to(gen_data.gpu_device)
    dist.all_reduce(my_val, op=dist.ReduceOp.SUM)
    return(my_val.item())

def my_print(outstr):
    '''
    Routine for printing by any processor/gpu.

            Parameters:
                    outstr (str): string to print.
    '''
    print('[%d]: %s' % (gen_data.my_rank, outstr))

def master_print(outstr):
    '''
    Routine for printing by rank 0  processor/gpu.

            Parameters:
                    outstr (str): string to print.
    '''
    if(gen_data.my_rank == 0):
        print('[MASTER]: %s' % (outstr))

def barr_time():
    '''
    Routine for barrier implementation across processors/gpus.

            Returns:
                    Time at end of the barrier.
    '''
    dummy = torch.tensor([0]).to(gen_data.gpu_device)
    dist.all_reduce(dummy, op=dist.ReduceOp.SUM)
    #dist.all_reduce(dummy, op=dist.reduce_op.SUM)
    return(time.time())

def dist_setup(num_ranks=None, my_rank=None, gpu_device=True):
    '''
    Routine for setting up distributed environment.
    Updates gen_data object with num_ranks, my_rank and gpu_device
    '''
    if(gpu_device==False):
        gen_data.num_ranks = 1
        gen_data.my_rank = 0
        return
    if(num_ranks==None):
        num_ranks = int(os.environ['SLURM_NTASKS'])
    if(my_rank==None):
        my_rank = int(os.environ['SLURM_PROCID'])

    if my_rank == 0:
        f = open("master_ip.txt", "w")
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        f.write(ip_address)
        f.close()
    else:
        time.sleep(5)
        f = open("master_ip.txt", "r")
        ip_address = f.read()
    os.environ['MASTER_ADDR'] = ip_address
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=my_rank, world_size=num_ranks)
    gen_data.num_ranks = num_ranks
    gen_data.my_rank =  my_rank
    gen_data.gpu_device = torch.device('cuda:' + str(my_rank % 8))
    print('Connected Sucessfully', my_rank, num_ranks, ip_address, gen_data.gpu_device)
    if os.path.exists("master_ip.txt"):
        os.remove("master_ip.txt")

def GPU_mem():
    '''
    Routine for reporting gpu memory used.

            Returns:
                    Maximum gpu memory used for current rank/process.
    '''
    device = gen_data.gpu_device
    return(int(torch.cuda.max_memory_allocated(device)/(1000*1000)))

def show_hist(arr):
    '''
    Routine for printing histogram of a tensor.

            Parameter:
                   arr (tensor): input tensor on which histogram is to be printed.
    '''
    arr = arr.detach().cpu().numpy()
    h = np.histogram(arr)
    print(h[0])
    print('')
    print(h[1])


def setup_values(gpu_device, nepochs, lr, optimizer):
    gen_data.config = {}
    gen_data.config['gpu_device'] = bool(gpu_device)
    gen_data.config['nepochs'] = int(nepochs)
    gen_data.config['lr'] =  float(lr)
    gen_data.config['optimizer'] = optimizer

    gen_data.config['smooth_alpha'] = 50.0
    gen_data.config['w_clamp_thr_lb'] = 0.0
    gen_data.config['w_clamp_thr_ub'] = 1.0
    gen_data.config['bias_clamp_thr_lb'] = -6.0
    gen_data.config['bias_clamp_thr_ub'] = 6.0
    gen_data.config['cpu_gpu_swap'] = False
    gen_data.config['eps'] = 0.01
    gen_data.config["inf_steps"] = 1
    if(gen_data.config['gpu_device']): gen_data.gpu_device = torch.device('cuda:0')
    else: gen_data.gpu_device = torch.device('cpu')




class timer:
    flag = False
    fwd_dict = {
            'total' : 0,
            'coal' : 0,
            'col_sum' : 0,
            'misc'     : 0, 
    }
    bak_dict = {
            'total' : 0,
            'coal' : 0,
            'col_sum' : 0,
            'misc'     : 0, 
    }
    grad_dict = {
            'total' : 0,
            'coal' : 0,
            'misc'  : 0
    }
    gen_dict = {
            'total' : 0,
            'bak_grad': 0,
            'misc'  : 0
    }

def time_start():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return start,end

def time_end(start, end, dir, comp):
    end.record()
    torch.cuda.synchronize()
    ttime = start.elapsed_time(end)
    ttime = int(ttime)
    if(dir == None): return(ttime)
    assert(dir in {'fwd', 'bak', 'grad', 'gen'})
    if(timer.flag):
        if(dir == 'fwd'):
            assert(comp in timer.fwd_dict)
            timer.fwd_dict[comp] += ttime
        if(dir == 'bak'):
            assert(comp in timer.bak_dict)
            timer.bak_dict[comp] += ttime
        if(dir == 'grad'):
            assert(comp in timer.grad_dict)
            timer.grad_dict[comp] += ttime
        if(dir == 'gen'):
            assert(comp in timer.gen_dict)
            timer.gen_dict[comp] += ttime
    return(ttime)

def print_timer():
    d = timer.fwd_dict
    d['misc'] = d['total'] - d['coal'] - d['col_sum']
    d = timer.bak_dict
    d['misc'] = d['total'] - d['coal'] - d['col_sum']
    d = timer.grad_dict
    d['total'] = timer.gen_dict['bak_grad'] - timer.bak_dict['total'] 
    d['misc'] = d['total'] - d['coal']
    d = timer.gen_dict
    d['misc'] = d['total'] - (timer.fwd_dict['total'] + timer.bak_dict['total'] + timer.grad_dict['total'])

    print('** forward **')
    print(timer.fwd_dict)
    print('*** bak **')
    print(timer.bak_dict)
    print('*** grad **')
    print(timer.grad_dict)
    print('*** gen **')
    print(timer.gen_dict)

class gen_data:
    """
    A class for useful data structures.

    Attributes
    ----------
    config : input json file loaded here 
    dir :  checkpoint direction
    num_ranks : number of processes/gpus
    my_rank : rank of this process/ gpu
    cpu_device: torch cpu device
    gpu_device: torch gpu device id for my rank
    phase_print: bool for printing debug info
    """

    config = None  #input json file loaded here
    dir = None #checkpoint direction
    num_ranks = None
    my_rank = None
    cpu_device = torch.device('cpu')
    gpu_device = torch.device('cuda:0')
    phase_print = False
