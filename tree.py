from collections import namedtuple
#from random import choice, uniform
import random
from monte_carlo_tree_search import MCTS, Node
from parameters_MCTS_SO import*
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import layers

import itertools
import time

import os
os.environ['PYTHONHASHSEED']=str(seed_value)# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

_T = namedtuple("node", "tup terminal")

def quad_beg(cost):
    a = (coeff) / (total_cost) ** 2 # c = value at high cost, d = high cost (41), k,h=0
    y = a * (cost) ** 2
    return y

def linear_function(cost): #c = coeff, d = high cost, h = 0, k = 0
    a = coeff / total_cost
    y = a*cost
    return y

def quad_end(cost):
    a = -coeff / total_cost ** 2#-y_1/x_1 **2
    b = 2 * coeff / total_cost #2*y_1/x_1
    y = a * cost ** 2 + b * cost
    return y

def constant():
    return coeff

if cost_name == 'fit':
    fit_name = 'finalized_model_physionet_smote_fit_' + str(random_state) + '_' + str(seed_value) + '.sav'
    model_fit = pickle.load(open(fit_name, 'rb'))

_T = namedtuple("node", "tup terminal")

class Tree(_T, Node):

    def find_children(node,i,X_train):
        
        if node.terminal: 
            return set()

        lst = [[i] for i in range(0,784)]

        pairs = [[i,i] for i in range(0,784)]

        dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

        to_be_acquired = []
        for j in range(len(lst)):
            lst_to_be_checked = lst[j]
            if node.tup[lst_to_be_checked[0]] == -1:
                to_be_acquired.append(lst_to_be_checked[0])

        dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(to_be_acquired) if val in features]

        num_features = list(set(dct_indx))

        return {
            node.make(value,i,X_train) for j, value in enumerate(num_features)
        }

    def find_random_child(node,i,X_train):

        if node.terminal:
            return None 

        lst = [[i] for i in range(0,784)]

        pairs = [[i,i] for i in range(0,784)]

        dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

        to_be_acquired = []
        for j in range(len(lst)):
            lst_to_be_checked = lst[j]
            if node.tup[lst_to_be_checked[0]] == -1:
                to_be_acquired.append(lst_to_be_checked[0])

        dct_indx = [k for k, features in enumerate(lst) for j, val in enumerate(to_be_acquired) if val in features]

        num_features = list(set(dct_indx))

        return node.make(random.choice(num_features),i,X_train) #one random child node

    def reward(state,model_sk):

        cost = 0
        total_cost = 784.0

        lst = [[i] for i in range(0,784)]

        pairs = [[i,i] for i in range(0,784)]

        dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

        acquired = []
        for k in range(len(lst)):
            lst_to_be_checked = lst[k]
        #if lst_to_be_checked[0] <= 33 and tup[lst_to_be_checked[0]] != 1:
        #    acquired.append(lst_to_be_checked[0])
            if state.tup[lst_to_be_checked[0]] != -1:
                acquired.append(lst_to_be_checked[0])

        dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

        num_features = list(set(dct_indx))

        for i in range(len(num_features)):
            cost += 1

        state = np.asarray(state.tup).flatten()

        indx = [dict_lst[feature] for feature in dct_indx]
        flat_list = [item for sublist in indx for item in sublist]

        if cost_name == 'fit':
            if flat_list:
                if len(flat_list) < 54:
                    if tuple(flat_list) in model_fit:
                        model = model_fit[tuple(flat_list)]
                        state = np.array([val for i,val in enumerate(state) for j,index in enumerate(flat_list) if i == index])
                    else:
                        model = model_sk
                else:
                        model = model_sk
            else:
                model = model_sk
        else:
            model = model_sk
            
        if cost_name == 'quadratic':
            if further_name == 'end':
                state = np.array([quad_end(cost) if state[i] == -1.0 else state[i] for i in range(len(state))])
            else:
                state = np.array([quad_beg(cost) if state[i] == -1.0 else state[i] for i in range(len(state))])
        elif cost_name == 'linear':
            state = np.array([linear_function(cost) if state[i] == -1.0 else state[i] for i in range(len(state))])
        else:
            state = np.array([constant() if state[i] == -1.0 else state[i] for i in range(len(state))])

        # X = state

        # W = model.get_weights()
        # X      = X.reshape((-1,X.shape[0]))           #Flatten
        # X      = X @ W[0] + W[1]                      #Dense
        # X[X<0] = 0                                    #Relu
        # X      = X @ W[2] + W[3]                      #Dense
        # X[X<0] = 0                                    #Relu
        # X      = X @ W[4] + W[5]                      #Dense
        # X[X<0] = 0                                    #Relu
        # X      = X @ W[6] + W[7]                      #Dense
        # X[X<0] = 0                                    #Relu
        # X      = np.exp(X)/np.exp(X).sum(1)[...,None] #Softmax

        #model = model_sk
        state = state.reshape(1,-1)
        classification = model.predict(state) #0 or 1
        prob = model.predict_proba(state).flatten()
        classification = prob[classification[0]]
        #classification = np.max(X)
        cost = (cost + 1.0)/(total_cost + 1.0) #cost always increasing from 0 to 1
        return_value = classification/cost
        return_value = return_value.flatten()

        return return_value

    def is_terminal(node):
        return node.terminal

    def make(vec,action,i,X_train): #action from 0 to 15
        
        lst = [[i] for i in range(0,784)]

        pairs = [[i,i] for i in range(0,784)]

        dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

        val = tuple([X_train.iloc[i,dict_lst[action][j]] for j,value in enumerate(dict_lst[action])])

        tup = vec.tup[:dict_lst[action][0]] + val + vec.tup[dict_lst[action][-1]+1:]

        acquired = []
        for k in range(len(lst)):
            lst_to_be_checked = lst[k]
        #if lst_to_be_checked[0] <= 33 and tup[lst_to_be_checked[0]] != 1:
        #    acquired.append(lst_to_be_checked[0])
            if tup[lst_to_be_checked[0]] != -1:
                acquired.append(lst_to_be_checked[0])

        dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

        num_features = list(set(dct_indx))

        is_terminal = len(num_features) == 784
        
        return Tree(tup, is_terminal) #new node