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

    def reward(node,model_sk,X_train,i):

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

        for i in range(len(num_features)):
            cost += 16

        state = np.asarray(node.tup).flatten()

        #state = state.reshape(1,-1)
        #classification = model_sk.predict(state) #0 or 1
        #prob = model_sk.predict_proba(state).flatten()
        #classification = prob[classification[0]]
        state = state.reshape((-1,28,28,1))
        classification = model_sk(state,training=False)#.predict(state)
        classification = classification[0]
        classification = np.max(classification)#classification[Y_train[m]]#int(classification[0])]
        #classification = np.max(X)
        cost = (cost + 1.0)/(total_cost + 1.0) #cost always increasing from 0 to 1
        return_value = classification/cost
        return_value = return_value.flatten()

        return return_value

    def is_terminal(node):
        return node.terminal

    def make(node,action,i,X_train): #action from 0 to 15

        lst = [[i] for i in range(0,784)]

        pairs = [[i,i] for i in range(0,784)]

        dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

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