import os
import sys
import itertools


#lnn_rule_home = '/NFS/LNN/wnet-twc/'
lnn_home = '../../'
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

# utils for FOL LNN learning
def all_equal(iterator):
    if len(iterator)<=1:
        return False
    
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)

def get_all_combs(list_vars, num_arity=2, remove_duplicate=False):
    if remove_duplicate:
        return [x for x in itertools.product(*([list_vars]*num_arity)) if not all_equal(x)]
    else:
        return [x for x in itertools.product(*([list_vars]*num_arity))]
    
def get_output_arity(positive_data):
    op_arity={}
    for key in positive_data:
        op_arity[key]=len(positive_data[key][0])
    return op_arity

def get_all_objects(background_kg):
    # Gathers all the objects from the positive/negative data
    objects = []
    for k, v in background_kg.items():
        for item in v:
            objects+=list(item)
    objects = list(set(objects))
    return objects

def get_negative_samples(positive_data, all_objects, num_neg_samples=10):
    # Computes all posible combinations from the objects and removes positive data
    # Samples num_neg_samples number of negative samples
    N = {}
    arity_dict = get_output_arity(positive_data)
    for key in positive_data:
        all_groundings = get_all_combs(objects, num_arity=arity_dict[key], remove_duplicate=False)
        for x in positive_data[key]:
            all_groundings.remove(x)
        if num_neg_samples is not None:
            N[key]=random.sample(all_groundings, num_neg_samples)
        else:
            N[key]=all_groundings
    return N

def ground_predicate(x, pred_name, facts):
    return (x in facts[pred_name])




class FOLGrounding:
    def __init__(self, B, P, N=None, orig_vars=None, free_vars=None, use_negatives=True, num_neg_samples=None, remove_duplicate_free_vars=False):
        self.B = B
        self.P = P
        self.all_objects = get_all_objects(B)
        if N is None:
            if use_negatives:
                # assert num_neg_samples is not None, "Please specify the number of the negative samples to be used"
                self.N = get_negative_samples(self.P, self.all_objects, num_neg_samples=num_neg_samples)
            else:
                self.N={k:[] for k in self.P}
        else:
            self.N = N

        # The variables in the final rule
        # Orig vars should be in the same order as output var order
        self.orig_vars = orig_vars
        self.free_vars = sorted(free_vars)
        self.all_vars = orig_vars+sorted(free_vars)

        self.ip_pred_arity = get_output_arity(B)
        for k,v in get_output_arity(P).items():
            self.op_pred_name, self.op_pred_arity = k,v
        self.all_pred_combs = self.get_all_preds(remove_duplicate_free_vars=remove_duplicate_free_vars)

    def convert_to_template(self, pred_args):
        args = pred_args.split(',')
        vals=[]
        num_free_vars=0
        unique_vars=[]
        for x in args:
            cond = x in self.free_vars
            vals.append("<f>" if cond else x)
            num_free_vars+= 1 if cond else 0
            unique_vars.append(x)
        return ",".join(vals), num_free_vars, ",".join(sorted(unique_vars))

    def get_all_preds(self, remove_duplicate_free_vars=False):
        preds_templates = []
        for pred in self.B:
            all_possible = [",".join(x) for x in itertools.product(*([self.all_vars]*self.ip_pred_arity[pred])) if not all_equal(x)]
            # print(pred, all_possible)
            # import ipdb; ipdb.set_trace()
            if remove_duplicate_free_vars:
                duplicate_hash={}
                for item in all_possible:
                    template, num_free_vars, unique_vars=self.convert_to_template(item)
                    if unique_vars not in duplicate_hash:
                        preds_templates.append(pred+'('+item+')')
                        if num_free_vars>1:
                            duplicate_hash[unique_vars]=pred
            else:
                preds_templates += [pred+'('+item+')' for item in all_possible]

        return preds_templates

    def predicate_with_exist(self, relational_pred, groundings={'a': None, 'b': None}):
        facts = self.B
        possible_zs = self.all_objects
        #### Split the predicate template ####
        predname = relational_pred.split('(')[0]
        predargs = relational_pred.split('(')[-1].split(')')[0].split(',')

        free_vars_in_template = [x for x in self.free_vars if (x in predargs)]
        has_free_variable = len(free_vars_in_template)>0

        if not has_free_variable:
            ground_tuple = tuple([groundings[k] for k in predargs])
            return ground_predicate(ground_tuple, predname, facts)
        else:
            free_variable_combs = get_all_combs(possible_zs, len(free_vars_in_template), False)
            for item in free_variable_combs:
                for v, k in zip(item, free_vars_in_template):
                    groundings[k]=v
                ground_tuple = tuple([groundings[k] for k in predargs])
                if ground_predicate(ground_tuple, predname, facts):
                    return True
            return False

    def compute_logic_vector(self, groundings):
        logic_vector=[]
        for rel in self.all_pred_combs:
            val = self.predicate_with_exist(relational_pred=rel, groundings=groundings)
            logic_vector.append(float(val))
        return logic_vector


    def get_train_data(self, pred2learn='isGrandFather', use_pos_data_only=False):
        all_pos_vecs = []
        for pairs in self.P[pred2learn]:
            groundings={}
            for v, k in zip(pairs, self.orig_vars):
                groundings[k]=v
            vec = self.compute_logic_vector(groundings)
            all_pos_vecs.append(vec)

        all_neg_vecs = []
        for pairs in self.N[pred2learn]:
            groundings={}
            for v, k in zip(pairs, self.orig_vars):
                groundings[k]=v
            vec = self.compute_logic_vector(groundings)
            all_neg_vecs.append(vec)

        pos_x = torch.from_numpy(np.array(all_pos_vecs))
        pos_y = torch.from_numpy(np.array([1.]*len(all_pos_vecs))).unsqueeze(-1)
        neg_x = torch.from_numpy(np.array(all_neg_vecs))
        neg_y = torch.from_numpy(np.array([0.]*len(all_neg_vecs))).unsqueeze(-1)

        if use_pos_data_only:
            full_X = torch.cat((pos_x,), 0).float()
            Y = torch.cat((pos_y, ), 0).float()
        else:
            full_X = torch.cat((pos_x, neg_x), 0).float()
            Y = torch.cat((pos_y, neg_y), 0).float()

        return full_X, Y


from tqdm import tqdm
class FOLGroundingCombined(FOLGrounding):
    def compute_logic_vector(self, groundings):
        logic_vector=[]
        facts = self.B
        possible_zs = self.all_objects
        free_variable_combs = get_all_combs(possible_zs, len(self.free_vars), len(self.free_vars)>1)
        # print(free_variable_combs)
        
        logic_vec_len_log = []
        for item in free_variable_combs:
            for v, k in zip(item, self.free_vars):
                groundings[k]=v
            logic_vector=[]
            for relational_pred in self.all_pred_combs:
                #### Split the predicate template ####
                predname = relational_pred.split('(')[0]
                predargs = relational_pred.split('(')[-1].split(')')[0].split(',')
        
                ground_tuple = tuple([groundings[k] for k in predargs])
                # print(relational_pred, predargs, ground_tuple, groundings)
                logic_vector.append(ground_predicate(ground_tuple, predname, facts))
            
            logic_vec_len_log.append([logic_vector, np.sum(logic_vector)])

        max_vec_len = np.max([l for _, l in logic_vec_len_log])
        best_logic_vec = [x for x, l in logic_vec_len_log if l==max_vec_len]
        return best_logic_vec
    
    def get_train_data(self, pred2learn='isGrandFather', use_pos_data_only=False):
        all_pos_vecs = []
        for pairs in tqdm(self.P[pred2learn]):
            groundings={}
            for v, k in zip(pairs, self.orig_vars):
                groundings[k]=v
            vec = self.compute_logic_vector(groundings)
            all_pos_vecs+=vec
            
        all_neg_vecs = []
        for pairs in tqdm(self.N[pred2learn]):
            groundings={}
            for v, k in zip(pairs, self.orig_vars):
                groundings[k]=v
            vec = self.compute_logic_vector(groundings)
            all_neg_vecs+=vec

        pos_x = torch.from_numpy(np.array(all_pos_vecs))
        pos_y = torch.from_numpy(np.array([1.]*len(all_pos_vecs))).unsqueeze(-1)
        neg_x = torch.from_numpy(np.array(all_neg_vecs))
        neg_y = torch.from_numpy(np.array([0.]*len(all_neg_vecs))).unsqueeze(-1)
        
        if use_pos_data_only:
            full_X = torch.cat((pos_x,), 0).float()
            Y = torch.cat((pos_y, ), 0).float()
        else:
            full_X = torch.cat((pos_x, neg_x), 0).float()
            Y = torch.cat((pos_y, neg_y), 0).float()
            
        return full_X, Y



def learn_lnn_rule_free_var(full_X, Y, all_pred_replaced, pred2learn, constrained=True, 
                            use_lambda=True, lam=0.0001, check_constraints=True, th=0.95):

    num_inputs=full_X.size(1)

    gpu_device = True
    nepochs = 500
    lr = 0.05
    optimizer = "SGD"
    op = tensorlnn.NeuralNet(num_inputs, gpu_device, nepochs, lr, optimizer)

    print('Samples before training...')
    #_ = op.display_weights(normed=True)
    # optimizer = optim.Adamax(op.parameters(), lr=0.01, weight_decay=1e-5)
    #optimizer = optim.Adam(op.parameters(), lr=0.01, weight_decay=1e-5)
    #loss_fn = nn.BCELoss()  # nn.BCEWithLogitsLoss()

    print(full_X)
    print()
    print(Y)

    t1 = time.time()

    op.train(full_X, Y)

    '''
    for iter_ in range(500):
        op.train()
        optimizer.zero_grad()
        yhat, sum_slacks = op(full_X)
        loss = loss_fn(yhat, Y) #turn this on if using BCEWithLogitsLoss
        if use_lambda:
            # print("\n #####", " Adding constrained loss ", "##### \n")
            constrained_loss = op.and_node.compute_constraint_loss(lam=lam) if op.and_node.lam else 0.0
            loss = loss + constrained_loss
        if (iter_+1)%100==0:
            print("Iteration " + str(iter_) + ": " + str(loss.item()))
        loss.backward()
        optimizer.step()
    '''    
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
    learned_pos_wts = [all_pred_replaced[k] for k,x in enumerate(wts) if x>th]
    pos_preds = list(set(learned_pos_wts))
    #learned_rule = "{}(x,y):- ".format(pred2learn) + " ∧ ".join(pos_preds)
    learned_rule = "{}(x,y):- ".format(pred2learn) + " and ".join(pos_preds)
    print('\n\n'+'##'*30)
    print("Learned rule : ")
    print(learned_rule)
    print('##'*30+'\n\n')
    
    '''
    if check_constraints:
        print('##'*30)
        print('# Constraint Check: #')
        check_constraints = op.check_constraint()
        violation = np.sum([max(x[0]-x[1], 0) for x in check_constraints])
        print('Constraint viloation: ', violation)
        if violation<1e-5:
            print('Result: Constraints are matching.')
        else:
            print('Result: Constraints are not matching.')
        print('##'*30)
    '''
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
#    pred = (yhat.squeeze().detach().numpy()>th).astype('float32')
#     pred = (yhat.squeeze().detach().numpy()).astype('float32')

#     print(true)
#     print(pred)

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
    inter = [all_pred_combs[j] for j in inter]
    print('Inter', inter)
    return X

import random

#with open("/NFS/LNN/wnet-twc/config.json", "r") as read_file:
#    util.gen_data.config = json.load(read_file)
util.dist_setup(1, 0)

B = {'atlocation': [('hat', 'rack'), ('moccasins', 'cabinet'), ('tissue', 'basket')],
     'related_to': [('shoe cabinet', 'cabinet'), ('hat rack', 'rack'), ('top hat', 'hat'), ('blue moccasins', 'moccasins'),
                    ('used tissue', 'tissue'), ('wastepaper basket', 'basket')]}
objects = get_all_objects(B)
P = {'put': [('top hat', 'hat rack'), ('blue moccasins', 'shoe cabinet'), ('used tissue', 'wastepaper basket')]}

foldata_actor = FOLGroundingCombined(B, P, N=None, orig_vars=['x', 'y'], free_vars=['u', 'v'], use_negatives=True, num_neg_samples=None, remove_duplicate_free_vars=True)
full_X, Y = foldata_actor.get_train_data(pred2learn='put', use_pos_data_only=True)

#full_X = random_fullX(full_X, foldata_actor.all_pred_combs)

op = learn_lnn_rule_free_var(full_X, Y, foldata_actor.all_pred_combs, pred2learn='put', constrained=True, use_lambda=True, lam=0.00001, check_constraints=False, th=0.1)


infer_lnn_rule(full_X, Y, op)

print(foldata_actor.all_pred_combs)
print("number of predicates:", len(foldata_actor.all_pred_combs))



