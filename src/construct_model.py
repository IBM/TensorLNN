import torch
import torch.distributed as dist
import time
import numpy as np
import sys
import os
import preprocess
import util
import lnn_wsd
import gates

def canonize(G, wptr):
    '''
    Routine to coalesce G and accordingly permute wptr. 
    Assumptions: no duplicate indices in G._indices()
            Parameters:
                    G : Sparse Tensor.
                    wptr: Mapping of each edge to a global index.
            Returns:
                    G : Coalesced G. 
                    wptr: Appriately permuted wptr to map each edge to global index.

    Steps :
         1. Create an auxillary sparse tensor tempG with G._indices() as its indices and wptr as its values.
         2. Coalesce tempG.
         3. Return coalesced form of G and tempG._values().
    '''
    assert(G._nnz() == wptr.numel())
    tempG = torch.sparse.FloatTensor(G._indices(), wptr, G.size())
    tempG = tempG.coalesce()
    wptr = tempG._values()
    G = G.coalesce()
    return(G, wptr)

def graph_concat(Glist):
    '''
    Routine for graph concatenation of the elements of a list.
            Parameters:
                    Glist : list of sparse tensors. 
            Returns:
                    newG : sparse tensor obtained by concatenating the graphs in G_list.
   
    Explanation: 
            For simplicity assume G_list has two sparse tensors  A of size (n1 x n1) and B of size (n2 x n2)
            with m1 and m2 non-zero values respectively.
            Then the routine will return a sparse tensor of size ((n1+n2) x (n1+n2)) with (m1+m2) non-zero values.
            The first m1 non-zero values of this tensor will correspond to those of A and shall have the same 
            (row, column) indices as those in A. The last m2 non-zero values of this tensor will correspond 
            to those of B and shall have the (row, column) indices increased by (n1, n2) from the (row, column)
            values in B.
    '''
    # n_sum will contain the sum of the dimension for the sparse tensors in Glist
    n_sum = 0
    new_index_list = []
    new_val_list = []
    
    for G in Glist:
        n = G.size()[0]
        index = G._indices()
        # increment index by n_sum
        new_index = n_sum + index 
        new_val = G._values()
        new_index_list.append(new_index)
        new_val_list.append(new_val)
        n_sum += n

    new_index = torch.cat(new_index_list, dim=1)
    new_val = torch.cat(new_val_list, dim=0)
    newG = torch.sparse.FloatTensor(new_index, new_val, size=(n_sum, n_sum))
    return(newG)

def construct_mega_univ_from_locals(lAndG_list, and_wptr_list, and_bptr_list, lNotG_list, not_wptr_list, not_bptr_list, lBounds_list):
    '''
    Routine to construct mega universe from universe data structures.
            Parameters:
                    lAndG_list : list of sparse tensors for AndNet of universes. 
                    and_wptr_list : list of and_wptr for universes.
                    and_bptr_list : list of and_bptr for universes.
                    lNotG_list : list of sparse tensors for NotNet of universes. 
                    not_wptr_list : list of not_wptr for universes.
                    not_bptr_list : list of not_bptr for universes.
                    lBounds_list : list of bound nodes for universes.
            Returns:
                    mega_univ : mega universe constructed from the above universe data structures. 

    Steps :
         1. For lAndG_list and lNotG_list,  do graph concatenation of their members using subroutine graph_concat.
         2. For the wptr and bptr lists, do tensor concatenation of their members.
         3. Create bound tensors L and U for each universe, initialize L=0 and U=1. Set L=1 for the nodes defined in bounds.
            Do tensor concatenation of bound tensors across universes.
         4. Return mega universe object using classes defined in lnn_wsd.py
    '''

    # do graph concatenation of the elements of lAndG_list and lNotG_list.
    mega_lAndG = graph_concat(lAndG_list).coalesce().to(util.gen_data.cpu_device)
    mega_lNotG = graph_concat(lNotG_list).coalesce().to(util.gen_data.cpu_device)

    # do tensor concatenation of elements of and_wptr_list, and_bptr_list, not_wptr_list and not_bptr_list.
    mega_and_wptr = torch.cat(and_wptr_list).to(util.gen_data.cpu_device)
    mega_and_bptr = torch.cat(and_bptr_list).to(util.gen_data.cpu_device)
    mega_not_wptr = torch.cat(not_wptr_list).to(util.gen_data.cpu_device)
    mega_not_bptr = torch.cat(not_bptr_list).to(util.gen_data.cpu_device)

    # create AndNet and NotNet objects using the concatenated data structures obtained above.
    mega_and_net = gates.AndNet(mega_lAndG, mega_and_wptr, mega_and_bptr, squeeze=False)
    mega_not_net = gates.NotNet(mega_lNotG, mega_not_wptr, mega_not_bptr, squeeze=False)

    # define bound lists.
    L0_list = []
    U0_list = []

    for (lAndG, lBounds) in zip(lAndG_list, lBounds_list):
        n = lAndG.size()[0]

        # for a particular universe, create bound tensors L and U each of dimension as the 
        #     number of nodes in the universe. Initialize L=1, U=1 for the nodes defined
        #     in bounds for that universe and L=0, U=1 for the ones not defined.
        L0 = torch.zeros(n, device=util.gen_data.gpu_device)
        U0 = torch.ones(n, device=util.gen_data.gpu_device)
        L0[lBounds] = 1.0

        L0_list.append(L0)
        U0_list.append(U0)

    # do tensor concatenation of elements of L0_list and U0_list.
    mega_L0 = torch.cat(L0_list).to(util.gen_data.cpu_device)
    mega_U0 = torch.cat(U0_list).to(util.gen_data.cpu_device)

    # create mega universe object from the AndNet object, NotNet objects and mega bounds.
    mega_univ = lnn_wsd.Univ(mega_and_net, mega_not_net, mega_L0, mega_U0)

    return(mega_univ)

def construct_mega_univ(gLNN, univ_ids, learn_m=None):
    '''
    Routine to construct a mega universe for univ_ids, a group  of universes.
            Parameters:
                    gLNN : Sparse Tensor for global LNN.
                    univ_ids : group  of universes to be combined for creating mega universe
                    learn_m : Number of learnable edges.
            Returns:
                    mega_univ : mega universe obtained by combining universes.

    Steps :
         1. For every universe in univ_ids, define data structures for its ANDNet, NOTnet and bounds. 
            These are obtained by invoking construct_single_univ defined in preprocess.py
         2. Coalesce lAndG and lNotG so that their indices are in appropriate order.
         3. Create individual lists for each data structure by combining across universes.
         4. Construct mega universe from the different lists obtained above.

    '''
    gAndG_list = []
    and_wptr_list = []
    and_bptr_list = []
    gNotG_list = []
    not_wptr_list = []
    not_bptr_list = []
    lBounds_list = []

    for univ_id in univ_ids:
        # determine data structures for the ANDNet, NOTnet and bounds of the universe. 
        lAndG, and_wptr, and_bptr, lNotG, not_wptr, not_bptr, lBounds = preprocess.construct_single_univ(gLNN, univ_id)

        # coalesce lAndG and lNotG so that their indices are in appropriate order.
        lAndG, and_wptr = canonize(lAndG, and_wptr)
        lNotG, not_wptr = canonize(lNotG, not_wptr)

        # append the data structures to respective lists across mulitple universes.
        gAndG_list.append(lAndG)
        and_wptr_list.append(and_wptr)
        and_bptr_list.append(and_bptr)
        gNotG_list.append(lNotG)
        not_wptr_list.append(not_wptr)
        not_bptr_list.append(not_bptr)
        lBounds_list.append(lBounds)

    # construct mega universe from the different lists obtained above.
    mega_univ = construct_mega_univ_from_locals(gAndG_list, and_wptr_list, and_bptr_list, gNotG_list, not_wptr_list, not_bptr_list, lBounds_list)
    mega_univ.print_stats()
    return(mega_univ)

def univ_multiplier(univ_ids, n_univ, times):
    '''
    Routine to synthetically repeat a number of times to enable scaling experiments.
            Parameters:
                    univ_ids : list of universe ids 
                    n_univ (int): number of universes to consider from univ_ids
                    times (int): number of times each universe will be repeated.
            Returns:
                    univ_ids : list of universe ids after replication.

    '''
    if(n_univ == None): return(univ_ids)
    univ_ids = univ_ids[:n_univ]
    univ_ids = univ_ids * times
    return(univ_ids)

def construct_wnet_model(gLNN, learn_m):
    '''
    Routine to construct the WordNet model. 
            Parameters:
                    
                    gLNN : Sparse Tensor.
                    learn_m : Number of learnable edges.
            Returns:
                    wnet_model: Object of WordNet class defined in lnn_wsd.py
    Steps :
         1. Read the universes.txt and determine the universes assigned to this process/gpu.
         2. Partition the universes assigned into groups, each group have a number of universes as given in config.
         3. Create my_mega_univ_list, a list of mega universes by constructing mega universe for each group.
         4. Create wnet_model from input parameters and the my_mega_univ_list.

    '''

    prep_st = time.time()

    # determine my rank, config parameters and gLNN dimensions
    gen_data = util.gen_data
    num_ranks = gen_data.num_ranks
    my_rank = gen_data.my_rank
    config = gen_data.config
    gN = gLNN.size()[0]
    gm = gLNN._nnz()
    util.my_print('-> Global LNN: gN = %d gm = %d' % (gN, gm))

    # read the universes.txt and determine the universes assigned to this process/gpu
    univ_list_fname = os.path.join(util.gen_data.config['univ_dir'], 'universes.txt')
    univ_ids = list(np.loadtxt(univ_list_fname,dtype=int))

    # provision to repeat universes a number of times to enable scaling study.
    univ_ids = univ_multiplier(univ_ids, 16, 1)
    n_univ = len(univ_ids)

    # split universes based on number of gpus/processes
    univ_split = np.array_split(np.array(univ_ids), num_ranks)

    # get universes assigned to this gpu/process
    my_univ_ids = univ_split[my_rank].tolist()
    my_n_univ = len(my_univ_ids)

    # partition the universes assigned into groups, each group have a number of universes as given in config
    group_size = config['group_size']
    my_n_groups = int(my_n_univ/group_size)
    univ_split = np.array_split(np.array(my_univ_ids), my_n_groups)
    my_groups = [g.tolist() for g in univ_split]

    util.my_print('---> My univ n_univ = %d my_n_univ = %d group_size = %d' % (n_univ, my_n_univ, group_size))

    # Create a list of mega-universes by constructing mega universe for each group g using subroutine construct_mega_univ
    my_mega_univ_list = []
    for g_num, g in enumerate(my_groups):
        util.my_print('-> Processing mega univ %d of %d' % (g_num, my_n_groups))
        mega_univ = construct_mega_univ(gLNN, g, learn_m)
        my_mega_univ_list.append(mega_univ)

    # Create wnet_model from input parameters and the my_mega_univ_list.
    wnet_model = lnn_wsd.WordNet(gLNN, learn_m, my_mega_univ_list)
    util.my_print('-> Preprocessing: Time %.1f GPU_mem = %d' % (time.time() - prep_st, util.GPU_mem()))
    return(wnet_model)
