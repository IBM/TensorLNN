import os
import sys
import itertools


#lnn_rule_home = '/NFS/LNN/wnet-twc/'
lnn_home = '../..'
lnn_src = '../../src'

sys.path.append(lnn_home)
sys.path.append(lnn_src)


import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import util
import tensorlnn

import json


def learn_lnn_rule_free_var(full_X, Y, th=0.95):

    num_inputs=full_X.size(1)

    nepochs = 500
    lr = 0.05
    optimizer = "SGD"
    op = tensorlnn.NeuralNet(num_inputs, gpu_device, nepochs, lr, optimizer)

    print('Training Samples before training...')

    print(full_X)
    print()
    print(Y)

    t1 = time.time()

    op.train(full_X, Y)

    t2 = time.time()
    
    total_time = t2 - t1
    print("\n\n Time required for training: ", total_time, "sec")
    print("\n")
    
    print('After training...')
    #beta, wts = op.display_weights(normed=True)
    beta, wts = op.display_weights()
    wts = wts.detach().numpy()

    # threshold the weights and 
    th=th
    learned_pos_wts = [k for k,x in enumerate(wts) if x>th]
    pos_preds = list(set(learned_pos_wts))

    print('True predicates : ')
    print(pos_preds)

    
    return op



def evaluate(pred, target):
    '''
    Computes precision, recall, f1 given prediction and ground truth
    labels. Meant for binary-class inputs, labels are 0 and 1.
    Parameters:
    ----------
    pred: vector of predicted labels
    target: vector of ground truth labels
    Returns: True-positives, False-positives, False-negatives,
    Precision, Recall and F1, in that order.
    '''
    tp = np.sum(pred * target)
    fp = np.sum(pred * (1-target))
    fn = np.sum((1-pred) * target)
    acc = np.sum(pred == target)/len(pred)

    #0/0 for precision/recall is defined as 1
    precision = tp / (tp + fp) if tp+fp > 0 else 1
    recall = tp / (tp + fn) if tp+fn > 0 else 1
    f1 = 2 * tp / (2*tp + fp + fn)

    return tp, fp, fn, precision, recall, f1, acc




def infer_lnn_rule(full_X, Y, op, th=0.9):
    # ToDo: inference needs to be on unseen data...
    #op.eval()
    #yhat, sum_slacks = op(full_X)

    lb, ub, _ = op.infer(full_X)
    true = Y.squeeze().detach().numpy()
    lb = lb.squeeze().detach().numpy()
    ub = ub.squeeze().detach().numpy()

    eps = 0.005
    if(max(abs(ub-lb)) > eps):
        print('Did not converge')
        return

    pred = (ub > th).astype('float32')

    tp, fp, fn, precision, recall, f1, acc = evaluate(true, pred)
    print('Precision :', precision)
    print('Recall :', recall)
    print('F1 :', f1)
    print('Accuracy :', acc)

def random_fullX(full_X, all_pred_combs):
    X = torch.randint(0, 2, full_X.shape).float()
    inter, _ = torch.min(X, 0)
    inter = inter.nonzero().squeeze()
    inter = inter.tolist()
    return X

import random

gpu_device = True
util.dist_setup(1, 0, gpu_device)
try:
    num_samples = int(sys.argv[1])
    num_nodes = int(sys.argv[2])
except:
    print("Command to run the code is `python basic_lnn.py <num_samples> <num_nodes>`.")
    exit(0)

if(num_samples < 1 or num_nodes <2):
    print("num_samples should be atleast 1 and num_nodes should be atleast 2.")
    exit(0)


full_X = torch.randint(0, 2, torch.Size([num_samples, num_nodes])).float()
Y = torch.ones([num_samples,1])

op = learn_lnn_rule_free_var(full_X, Y, th=0.1)


infer_lnn_rule(full_X, Y, op)




