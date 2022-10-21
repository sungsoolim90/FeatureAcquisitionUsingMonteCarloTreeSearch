from collections import namedtuple
import random
#from random import choice, uniform
from monte_carlo_tree_search_mo_mnist import MCTS, Node

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import layers

import numpy as np
import pickle

import itertools
import time

from parameters_MCTS_MO import*

import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

_T = namedtuple("node", "tup terminal")

class Tree(_T, Node):

    def find_children(node,i,X_train):
        
        if node.terminal: 
            return set()

        tup = np.array(node.tup)

        tup = tup.reshape((-1,28,28,1))

        to_be_acquired = []

        for k in range(49):
            row_action = int(k/7.0) ##row action
            column_action = int(k - row_action*7.0) ##
            if np.allclose(tup[:,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)],X_train[i,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)]):#,atol=1.0e-6):
                to_be_acquired.append(k) #already acquired

        num_features = list(set([i for i in range(49)]) - set(to_be_acquired))

        return {
            node.make(value,i,X_train) for j, value in enumerate(num_features)
        }

    def find_random_child(node,i,X_train):

        if node.terminal:
            return None 

        tup = np.array(node.tup)

        tup = tup.reshape((-1,28,28,1))

        lst = [[i] for i in range(0,784)]

        pairs = [[i,i] for i in range(0,784)]

        dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

        to_be_acquired = []

        for k in range(49):
            row_action = int(k/7.0) ##row action
            column_action = int(k - row_action*7.0) ##
            if np.allclose(tup[:,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)],X_train[i,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)]):#,atol=1.0e-6):
                to_be_acquired.append(k) #already acquired

        num_features = list(set([i for i in range(49)]) - set(to_be_acquired))

        action = random.choice(num_features)

        return node.make(action,i,X_train)#, action #one random child node

    def cost(node,i,X_train):

        cost = 0
        total_cost = 784.0

        tup = np.array(node.tup)

        tup = tup.reshape((-1,28,28,1))

        lst = [[i] for i in range(0,784)]

        pairs = [[i,i] for i in range(0,784)]

        dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

        to_be_acquired = []

        for z in range(49):
            row_action = int(z/7.0) ##row action
            column_action = int(z - row_action*7.0)
            if np.allclose(tup[:,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)],X_train[i,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)]):#,atol=1.0e-6):
                to_be_acquired.append(z) #already acquired

        num_features = list(set(to_be_acquired))

        for m in range(len(num_features)):
            cost += 16
                
        return cost/total_cost

    def f1(node,model_zero,y_train,j,X_train):

        cost = 0
        total_cost = 784.0

        tup = np.array(node.tup)

        tup = tup.reshape((-1,28,28,1))

        lst = [[i] for i in range(0,784)]

        pairs = [[i,i] for i in range(0,784)]

        dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

        to_be_acquired = []

        for z in range(49):
            row_action = int(z/7.0) ##row action
            column_action = int(z - row_action*7.0)
            if np.allclose(tup[:,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)],X_train[j,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)]):#,atol=1.0e-6):
                to_be_acquired.append(z) #already acquired

        num_features = list(set(to_be_acquired))

        for i in range(len(num_features)):
            cost += 16

        state = np.asarray(node.tup).flatten()

        # X = state

        # if cost_name == 'fit':
        #     if flat_list:
        #         W = model.get_weights()
        #         X      = X.reshape((-1,X.shape[0]))           #Flatten
        #         X      = X @ W[0] + W[1]                      #Dense
        #         X[X<0] = 0                                    #Relu
        #         X      = X @ W[2] + W[3]                      #Dense
        #         X[X<0] = 0                                    #Relu
        #         X      = X @ W[4] + W[5]                      #Dense
        #         X[X<0] = 0                                    #Relu
        #         X      = X @ W[6] + W[7]                      #Dense
        #         X[X<0] = 0                                    #Relu
        #         X      = np.exp(X)/np.exp(X).sum(1)[...,None] #Softmax
        #     else:
        #         W = model.get_weights()
        #         X      = X.reshape((-1,X.shape[0]))           #Flatten
        #         X      = X @ W[0] + W[1]                      #Dense
        #         X[X<0] = 0                                    #Relu
        #         X      = X @ W[2] + W[3]                      #Dense
        #         X[X<0] = 0                                    #Relu
        #         X      = X @ W[4] + W[5]                      #Dense
        #         X[X<0] = 0                                    #Relu
        #         X      = np.exp(X)/np.exp(X).sum(1)[...,None] #Softmax
        # else:
        #     W = model.get_weights()
        #     X      = X.reshape((-1,X.shape[0]))           #Flatten
        #     X      = X @ W[0] + W[1]                      #Dense
        #     X[X<0] = 0                                    #Relu
        #     X      = X @ W[2] + W[3]                      #Dense
        #     X[X<0] = 0                                    #Relu
        #     X      = X @ W[4] + W[5]                      #Dense
        #     X[X<0] = 0                                    #Relu
        #     X      = np.exp(X)/np.exp(X).sum(1)[...,None] #Softmax
    
        #classification = np.max(X)
        #classification = model.predict(tup) #0 or 1
        state = state.reshape(1,-1)
        prob = model_zero.predict_proba(state).flatten()
        classification = prob[y_train[j]]
        return classification

    def is_terminal(node):
        return node.terminal

    def make(node,action,i,X_train): #action from 0 to 15

        row_action = int(action/7.0) ##row action

        column_action = int(action - row_action*7.0) ##

        vec = np.array(node.tup)

        vec = vec.reshape((-1,28,28,1))

        vec[:,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)] = X_train[i,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1),:]

        acquired = []
        for k in range(49):
            row_action = int(k/7.0) ##row action
            column_action = int(k - row_action*7.0) ##
            if np.allclose(vec[:,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)],X_train[i,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)]):#,atol=1.0e-6):
                acquired.append(k) #already acquired

        is_terminal = len(acquired) == 49

        vec = vec.reshape((-1,784))
        
        return Tree(tuple(vec[0]), is_terminal) #new node