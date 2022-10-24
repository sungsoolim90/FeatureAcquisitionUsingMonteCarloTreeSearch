"""
This script implements the node class object for training
1) find_children: find all children nodes of given node
2) find_random_child: find a random child node of given node
3) reward: find reward for a given acquisition action
4) make: make a new node given action

"""

from collections import namedtuple
from monte_carlo_tree_search import MCTS, Node
from parameters_MCTS import*

import numpy as np
import random

import os
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)# 2. Seed python built-in pseudo-random generator
np.random.RandomState(seed_value)# 3. Seed numpy pseudo-random generator

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

        state = np.array(node.tup).flatten()

        state = state.reshape((-1,28,28,1))

        to_be_acquired = []

        for z in range(49):
            row_action = int(z/7.0) ##row action
            column_action = int(z - row_action*7.0)
            if np.allclose(state[:,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)],X_train[i,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)]):#,atol=1.0e-6):
                to_be_acquired.append(z) #already acquired

        num_features = list(set(to_be_acquired))

        for i in range(len(num_features)):
            cost += 16

        try:
            state = state.reshape((-1,784))
            classification_prob = np.max(model_sk.predict_proba(state).flatten())
        except AttributeError:
            classification_prob = np.max(model_sk.predict(state)).flatten()

        cost = (cost + 1.0)/(total_cost + 1.0) #cost always increasing from 0 to 1
        return_value = classification_prob/cost
        return_value = return_value.flatten()

        return return_value

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