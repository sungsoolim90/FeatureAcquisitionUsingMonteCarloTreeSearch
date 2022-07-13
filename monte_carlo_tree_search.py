"""
Code adapted from:

https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1

"""

from abc import ABC, abstractmethod
from collections import defaultdict
import math
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

        if node not in self.children:
            max_child = node.find_random_child(i,X_train)
        else:
            max_child = max(self.children[node],key=score)
        
        return max_child

    # def update(self,node,next_node):
    #     self.action[node] = [node.tup.index(x,i) for i, (x,y) in enumerate(zip(node.tup,next_node.tup)) if x != y]

    def train(self,node,i,model_zero,X_train):
        #with Pool(cpu_count()) as pool:
        path = self._select(node)
        leaf = path[-1] #leaf node
        self._expand(leaf,i,X_train) #children according to transition probability function - from the leaf node, expand
        reward = self._simulate(leaf,i,model_zero,X_train) #simulate to the end (random children)
        self._backpropagate(path, reward) #backpropagate to the root (increase reward and N for each node by 1)
        return self.Q, self.N

    def train_mp(self,node,i,model_zero,X_train):
        with Pool(cpu_count()) as pool:
            result = pool.map_async(self.train(node,i,model_zero,X_train),range(self.training))
            result.wait()
            pool.terminate()

    def _select(self, node):
        path = []
        while True:

            path.append(node)

            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                #print('return')
                return path

            #there is no unexplored
            unexplored = self.children[node] - self.children.keys()

            if unexplored:
                n = unexplored.pop()
                path.append(n)
                #print('unexplored')
                return path

            node = self._uct_select(node)  # descend a layer deeper

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
        while True:

            reward += node.reward(model_zero)
            
            if node.is_terminal():
                #print('done')
                break
                #reward = node.reward(model_sk)
                #return reward

            #if num_node > num:
            #    break
            #    reward = node.reward(model_sk)

            node = node.find_random_child(i,X_train)
            #reward += node.reward(model_sk)

            #num_node += 1
        return reward

    def _backpropagate(self, path, reward):
        
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node]) #n = child

        log_N_vertex = math.log(self.N[node])

        N_vertex = self.N[node]

        def uct(n):
            return self.Q[n] / N_vertex + self.exploration_weight * math.sqrt(log_N_vertex / N_vertex)

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