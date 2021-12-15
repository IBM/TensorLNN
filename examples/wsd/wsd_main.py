import torch
import time
import sys
import os
import torch.cuda.profiler as profiler
import json

lnn_home = '../../'
lnn_src = '../../src'

sys.path.append(lnn_home)
sys.path.append(lnn_src)


import lnn_wsd
import util
import preprocess
import construct_model

def main():
    """Top Level Main Function. """

    # setup  distributed environment.
    util.dist_setup(1, 0)
    print('*** Welcome: I am rank %d out of %d' % (util.gen_data.my_rank, util.gen_data.num_ranks))

    # read config file to determine lnn generation and training parameters
    with open("config.json", "r") as read_file:
        util.gen_data.config = json.load(read_file)

    # construct global LNN gLNN; learn_m number of learnable parameters 
    gLNN, learn_m = preprocess.construct_gLNN()

    # generate model wnet_model using gLNN, local universes and learnable parameters 
    wnet_model = construct_model.construct_wnet_model(gLNN, learn_m)

    # call training routine on the wnet_model
    wnet_model.train()

main()
