"""
Code adapted from:

https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1

"""

from abc import ABC, abstractmethod
from collections import defaultdict
import math
import numpy as np
#from multiprocessing import Pool, cpu_count
#import random

class MCTS:
    
    def __init__(self,exploration_weight=1.0):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.action = defaultdict(int)   # action - policy
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight #constant for UCB

    def choose(self,node,i,X_train): #Choose the node with highest score
        
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        #fix expand
        #fix make
        #fix checking next action
        if node not in self.children:
            print('node not in children')
            print('choose random child')
            max_child = node.find_random_child(i,X_train)
        else:
            max_child = max(self.children[node],key=score) #if all children are unvisited, return first child
            # next_action = self.update(node,max_child,act)
            # if next_action:
            #     if len(next_action) == 1:
            #         #print('correct behavior')
            #         #print(next_action)
            #         #print(len(self.children[node]))
            #         action = next_action[0]
            #     else:
            #         print('next action longer than one')
            #         print(next_action)
            #         action = next_action[0]
            # else:
            #     print('choose random child')
            #     #print(next_action)
            #     #This comes out to be true
            #     #If all the children nodes are not yet visited, they will have scores of -inf
            #     #
            #     #print(np.array_equal(np.array(max_child.tup),np.array(node.tup)))
            #     max_child, action = node.find_random_child(i,X_train,act)
        return max_child#, action

    # def update(self,node,next_node,act_lst):
    #     act = []
    #     tup_node = np.array(node.tup).reshape((-1,28,28,1))
    #     tup_next_node = np.array(next_node.tup).reshape((-1,28,28,1))
    #     for j in range(49):
    #         row_action = int(j/7.0) ##row action
    #         column_action = int(j - row_action*7.0) ##
    #         #np.array_equal(A,B)
    #         if not np.array_equal(tup_node[:,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)],tup_next_node[:,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)]):
    #             #print(j)
    #             #print(act_lst)
    #             if j not in act_lst:
    #                 act.append(j)
    #     #self.action[node] = [node.tup.index(x,i) for i, (x,y) in enumerate(zip(node.tup,next_node.tup)) if x != y]
    #     return act

    def train(self,node,i,model_zero,X_train):
        path = self._select(node,i,X_train)
        #print(act_lst)
        leaf = path[-1] #leaf node
        self._expand(leaf,i,X_train) #children according to transition probability function - from the leaf node, expand
        reward = self._simulate(leaf,i,model_zero,X_train) #simulate to the end (random children)
        self._backpropagate(path, reward) #backpropagate to the root (increase reward and N for each node by 1)

    def _select(self,node,i,X_train):
        path = []
        #act_lst = act.copy()
        while True:

            path.append(node)

            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path#, act_lst

            #there is unexplored
            unexplored = self.children[node] - self.children.keys()

            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path#, act_lst

            node = self._uct_select(node)  # descend a layer deeper

            # if np.array_equal(np.array(new_node.tup),np.array(node.tup)):
            #     if node.terminal:
            #         return path, act_lst
            #     else:
            #         node, next_action = node.find_random_child(i,X_train,act_lst)
            #         act_lst.append(next_action)
            # else:

            # next_action = self.update(node,new_node,act_lst)

            # node = new_node

            # if len(next_action) == 1:
            #     act_lst.append(next_action[0])
            # else:
            #     print('next action longer than one')
            #     print(next_action)
            #     act_lst.append(next_action[0])

    def _expand(self,node,i,X_train):
        
        if node in self.children:
            return  # already expanded
        if node.terminal:
            return
        else:
            self.children[node] = node.find_children(i,X_train)

    def _simulate(self,node,i,model_zero,X_train):
        #num_node = 1
        reward = 0.0
        #action_lst = act.copy()
        while True:

            reward += node.reward(model_zero,X_train,i)
            
            if node.is_terminal():
                break

            node = node.find_random_child(i,X_train)
            #action_lst.append(action)

        return reward#, path

    def _backpropagate(self, path, reward):
        
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node]) #n = child

        #log_N_vertex = math.log(self.N[node])

        #N_vertex = self.N[node]

        def uct(n):
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(math.log(self.N[n]) / self.N[n])

        return max(self.children[node], key=uct)


class Node(ABC):

    @abstractmethod
    def find_children(self):
        "All possible successors of this state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node."
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True