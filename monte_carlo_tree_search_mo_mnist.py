"""

Code adapted from:

https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1

"""

#Problems 
# 1) lower costs (simulation higher than global P)
# 2) the HV approximation for lower costs are higher than needed
# 3) local node approximation is not correct (the costs and probabilities adding up higher than need be)

from abc import ABC, abstractmethod
from collections import defaultdict
import math
import random

from parameters_MCTS_MO import*

import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)

def dominates(row, candidateRow):
    return sum([row[x] > candidateRow[x] for x in range(len(row))]) == len(row) #Return True or False

def hypervolume(front):
    reference = [-1.0, 0.0]
    h = front[1] - reference[1] #accuracy 
    hv = (front[0] - reference[0])*h
    return hv

def hypervolume_lst(front):
    reference = [-1.0, 0.0]
    front = sorted(front, key=lambda x: x[0]) #sort by first objective (cost)
    hv = 0
    for i in range(len(front)):
        h = front[i][1] - reference[1] #accuracy 
        if i == 0:
            hv += (front[i][0] - reference[0])*h
        else:
            hv += (front[i][0] - front[i-1][0])*h
    return hv

def simple_cull(inputPoints, dominates):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break

    return paretoPoints, dominatedPoints

import collections

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

def flatten(x):
    if isinstance(x, collectionsAbc.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

class MCTS:
    
    def __init__(self, exploration_weight = 1.0):
        self.N = defaultdict(int)  # total visit count for each node
        self.global_P = [] #defaultdict(list)  # global Pareto Front
        self.local_P = defaultdict(list) #local Reward vectors at each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight #constant for UCB

    def choose(self,node,i,X_train): #Choose the node with highest score

        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        def score(n):
            if self.N[n] == 0:
                return float("inf")  #avoid unseen moves

            P = [x for x in self.global_P if x]

            P, dominated = simple_cull(P,dominates)

            P = list(P)

            for i in range(len(P)):
                P[i] = list(P[i])

            if isinstance(self.local_P[n][0],float):
                local = [i/(self.N[node] + 1) for i in self.local_P[node]]
            else: #list of lists
                local = []
                for j in range(len(self.local_P[node])):
                    local.append([i/(self.N[node] + 1) for i in self.local_P[node][j]])
            #if isinstance(self.local_P[n][0],float):
            #    local = self.local_P[n] / self.N[n]
            #else: #list of lists
            #    local = []
            #    for j in range(len(self.local_P[n])):
            #        local.append(tree.local_P[n][j] / tree.N[n])

            local = flatten(local)
            row = int(len(local)/2.0)
            col = 2
            local = [local[col*i : col*(i+1)] for i in range(row)]

            P = [P, local]
            P = flatten(P)
            row = int(len(P)/2.0)
            col = 2
            P = [P[col*i : col*(i+1)] for i in range(row)]

            score = hypervolume_lst(P)

            return score

        if node not in self.children:
            max_child = node.find_random_child(i,X_train)
            #print('random next action')
        else:
            max_child = max(self.children[node],key=score)
            #print(score(max_child))
        
        return max_child, score(max_child)

    def train(self,node,i,model_zero,X_train,y_train):
        path = self._select(node,model_zero,y_train,i,X_train)
        leaf = path[-1] #leaf node
        self._expand(leaf,i,X_train,model_zero,y_train) #from the leaf node, expand
        reward = self._simulate(leaf,i,model_zero,X_train,y_train) #simulate to the end (random children)
        self._backpropagate(path,reward)#,model_zero,y_train,i) #backpropagate to the root (increase reward and N for each node by 1)
        #self.local_global(path,model_zero,y_train,i)

    def _select(self,node,model_zero,y_train,i,X_train):
        path = []
        while True:

            path.append(node)

            #if not self.global_P:
            self.global_P.append([-node.cost(i,X_train), node.f1(model_zero,y_train,i,X_train)])

            P, dominated = simple_cull(self.global_P,dominates) # Pareto Optimal Global P

            P = list(P)

            for v in range(len(P)):
                P[v] = list(P[v])

            self.global_P = P

            if not self.local_P[node]:
                self.local_P[node] = []
                self.local_P[node].append(-node.cost(i,X_train))
                self.local_P[node].append(node.f1(model_zero,y_train,i,X_train))

                self.N[node] += 1

            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path

            #there is no unexplored
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()

                self.global_P.append([-n.cost(i,X_train),n.f1(model_zero,y_train,i,X_train)])

                P, dominated = simple_cull(self.global_P,dominates) # Pareto Opitmal Global P
                
                P = list(P)
                for u in range(len(P)):
                    P[u] = list(P[u])

                self.global_P = P

                if not self.local_P[n]:
                    self.local_P[n] = []
                    self.local_P[n].append(-n.cost(i,X_train))
                    self.local_P[n].append(n.f1(model_zero,y_train,i,X_train))

                    self.N[node] += 1

                path.append(n)

                return path

            #this path
            node = self._uct_select(node)  # descend a layer deeper
            #self.update(node,model_zero)

    def _expand(self,node,i,X_train,model_zero,y_train):
        if node in self.children:
            return  # already expanded - does the global P get updated here?
        if node.terminal:
            return # terminal - does the global P get updated here?
        else:
            self.children[node] = node.find_children(i,X_train)
            
            for node_children in self.children[node]:
                self.global_P.append([-node_children.cost(i,X_train),node_children.f1(model_zero,y_train,i,X_train)])
                if not self.local_P[node_children]:
                    self.local_P[node_children] = []
                    self.local_P[node_children].append(-node_children.cost(i,X_train))#,node_children.f1(model_zero,y_train,i)])
                    self.local_P[node_children].append(node_children.f1(model_zero,y_train,i,X_train))

                    self.N[node_children] += 1

            P, dominated = simple_cull(self.global_P,dominates)
            P = list(P)
            for z in range(len(P)):
               P[z] = list(P[z])

            self.global_P = P

    def _simulate(self,node,i,model_zero,X_train,y_train):
        
        reward = []

        t = 0

        while True:

            t += 1

            if reward:
                reward[0] -= node.cost(i,X_train) #from 0 to 1
                reward[1] += node.f1(model_zero,y_train,i,X_train)
            else:
                reward.append(-node.cost(i,X_train))
                reward.append(node.f1(model_zero,y_train,i,X_train))

            reward[0] /= t
            reward[1] /= t

            # if not self.local_P[node]:
            #     self.local_P[node] = []
            #     self.local_P[node].append(-node.cost())
            #     self.local_P[node].append(node.f1(model_zero,y_train,i))

            self.global_P = [x for x in self.global_P if x]

            if self.global_P:
                if isinstance(self.global_P[0],float):
                    dom = dominates(self.global_P,reward) #what if global_P is a list of lists
                    r_dom = dominates(reward,self.global_P)
                    if dom == True:
                        self.global_P = self.global_P
                        #self.global_P = [x for x in self.global_P if x]
                    elif r_dom == True:
                        self.global_P = reward
                        #self.global_P = [x for x in self.global_P if x]
                    else:
                        #self.global_P.append(reward)
                        self.global_P = [self.global_P, reward]
                        self.global_P = flatten(self.global_P)
                        row = int(len(self.global_P)/2.0)
                        col = 2
                        self.global_P = [self.global_P[col*i : col*(i+1)] for i in range(row)]
                        self.global_P = [list(x) for x in set(tuple(x) for x in self.global_P)]
                        #self.global_P = [x for x in self.global_P if x]
                else:
                    self.global_P = [list(x) for x in set(tuple(x) for x in self.global_P)]
                    dom = []
                    for j in range(len(self.global_P)):
                        dom.append(dominates(self.global_P[j],reward))

                    r_dom = []
                    for j in range(len(self.global_P)):
                        r_dom.append(dominates(reward,self.global_P[j]))

                    if any(dom) == False:
                        #self.global_P = self.global_P
                        for z in range(len(dom)):
                            if dom[z] == False:
                                self.global_P[z] = reward
                        self.global_P = [list(x) for x in set(tuple(x) for x in self.global_P)]
                        #self.global_P = [x for x in self.global_P if x]
                    elif any(r_dom) == True:
                        for z in range(len(r_dom)):
                            if r_dom[z] == True:
                                self.global_P[z] = reward
                        self.global_P = [list(x) for x in set(tuple(x) for x in self.global_P)]
                        #self.global_P = [x for x in self.global_P if x]
                    else:
                        #self.global_P.append(reward)
                        self.global_P = [self.global_P, reward]
                        self.global_P = flatten(self.global_P)
                        row = int(len(self.global_P)/2.0)
                        col = 2
                        self.global_P = [self.global_P[col*i : col*(i+1)] for i in range(row)]
                        self.global_P = [list(x) for x in set(tuple(x) for x in self.global_P)]
                        #self.global_P = [x for x in self.global_P if x]
            else:
                self.global_P = reward

            # P, dominated = simple_cull(self.global_P,dominates) #Pareto Optimal Global P

            # P = list(P)

            # for i in range(len(P)):
            #    P[i] = list(P[i])

            # self.global_P = P

            if node.is_terminal():

                self.global_P = [x for x in self.global_P if x]

                if self.global_P:
                    if isinstance(self.global_P[0],float):
                        dom = dominates(self.global_P,reward)
                        r_dom = dominates(reward,self.global_P)
                        if dom == True:
                            self.global_P = self.global_P
                            #self.global_P = [x for x in self.global_P if x]
                        elif r_dom == True:
                            self.global_P = reward
                            #self.global_P = [x for x in self.global_P if x]
                        else:
                            #self.global_P.append(reward)
                            self.global_P = [self.global_P, reward]
                            self.global_P = flatten(self.global_P)
                            row = int(len(self.global_P)/2.0)
                            col = 2
                            self.global_P = [self.global_P[col*i : col*(i+1)] for i in range(row)]
                            self.global_P = [list(x) for x in set(tuple(x) for x in self.global_P)]
                            #self.global_P = [x for x in self.global_P if x]
                    else:
                        self.global_P = [list(x) for x in set(tuple(x) for x in self.global_P)]
                        dom = []
                        for j in range(len(self.global_P)):
                            dom.append(dominates(self.global_P[j],reward))

                        r_dom = []
                        for j in range(len(self.global_P)):
                            r_dom.append(dominates(reward,self.global_P[j]))

                        if any(dom) == False:
                            #self.global_P = self.global_P
                            for z in range(len(dom)):
                                if dom[z] == False:
                                    self.global_P[z] = reward
                            self.global_P = [list(x) for x in set(tuple(x) for x in self.global_P)]
                            #self.global_P = [x for x in self.global_P if x]
                        elif any(r_dom) == True:
                            for z in range(len(r_dom)):
                                if r_dom[z] == True:
                                    self.global_P[z] = reward
                            self.global_P = [list(x) for x in set(tuple(x) for x in self.global_P)]
                            #self.global_P = [x for x in self.global_P if x]
                        else:
                            #self.global_P.append(reward)
                            self.global_P = [self.global_P, reward]
                            self.global_P = flatten(self.global_P)
                            row = int(len(self.global_P)/2.0)
                            col = 2
                            self.global_P = [self.global_P[col*i : col*(i+1)] for i in range(row)]
                            self.global_P = [list(x) for x in set(tuple(x) for x in self.global_P)]
                            #self.global_P = [x for x in self.global_P if x]
                else:
                    self.global_P = reward
            
                # P, dominated = simple_cull(self.global_P,dominates)

                # P = list(P)

                # for i in range(len(P)):
                #    P[i] = list(P[i])

                # self.global_P = P

                return reward

            node = node.find_random_child(i,X_train)

    def _backpropagate(self,path,reward):#,model_zero,y_train,i):

        #during back propagation, the local P can be higher than -1.0 and 1.0

        t = 0

        for node in reversed(path):

            t += 1

            self.N[node] += 1

            if self.local_P[node]:

                self.local_P[node][0] += reward[0] #For local P, backpropagation
                self.local_P[node][1] += reward[1]

                #self.local_P[node] = [z/t for z in self.local_P[node]]

            else:

                self.local_P[node] = reward

            # self.local_P[node] = [x for x in self.local_P[node] if x]

            # if self.local_P[node]:
            #     if isinstance(self.local_P[node][0], float):
            #         dom = dominates(self.local_P[node],reward)
            #         r_dom = dominates(reward,self.local_P[node])
            #         if dom == True:
            #             self.local_P[node] = self.local_P[node]
            #         elif r_dom == True:
            #             self.local_P[node] = reward
            #         else:
            #             self.local_P[node] = [self.local_P[node], reward]
            #             self.local_P[node] = flatten(self.local_P[node])
            #             row = int(len(self.local_P[node])/2.0)
            #             col = 2
            #             self.local_P[node] = [self.local_P[node][col*i : col*(i+1)] for i in range(row)]
            #             self.local_P[node] = [list(x) for x in set(tuple(x) for x in self.local_P[node])]
            #     else:
            #         self.local_P[node] = [list(x) for x in set(tuple(x) for x in self.local_P[node])]

            #         dom = []
            #         for j in range(len(self.local_P[node])):
            #             dom.append(dominates(self.local_P[node][j],reward))

            #         r_dom = []
            #         for j in range(len(self.local_P[node])):
            #             r_dom.append(dominates(reward,self.local_P[node][j]))

            #         if any(dom) == False:
            #             for z in range(len(dom)):
            #                 if dom[z] == False:
            #                     self.local_P[node][z] = reward
            #             self.local_P[node] = [list(x) for x in set(tuple(x) for x in self.local_P[node])]
            #         elif any(r_dom) == True:
            #             for z in range(len(r_dom)):
            #                 if r_dom[z] == True:
            #                     self.local_P[node][z] = reward
            #             self.local_P[node] = [list(x) for x in set(tuple(x) for x in self.local_P[node])]
            #         else:
            #             self.local_P[node] = [self.local_P[node], reward]
            #             self.local_P[node] = flatten(self.local_P[node])
            #             row = int(len(self.local_P[node])/2.0)
            #             col = 2
            #             self.local_P[node] = [self.local_P[node][col*i : col*(i+1)] for i in range(row)]
            #             self.local_P[node] = [list(x) for x in set(tuple(x) for x in self.local_P[node])]
            # else:
            #     self.local_P[node] = reward

    # def local_global(self,path,model_zero,y_train,i):
    #     #delete locals that are not dominated by global

    #     if isinstance(self.global_P[0],float):
    #         for node in path:
    #             local = []
    #             if isinstance(self.local_P[node][0],float):
    #                 local.append([i/(self.N[node] + 1) for i in self.local_P[node]])
    #             else: #list of lists
    #                 for j in range(len(self.local_P[node])):
    #                     local.append([i/(self.N[node] + 1) for i in self.local_P[node][j]])

    #             n_dom = []
    #             for k in range(len(local)):
    #                 n_dom.append(dominates(local[k],self.global_P))

    #             true_indx = [i for i, x in enumerate(n_dom) if x]
    #             if true_indx:
    #                 for index in sorted(true_indx, reverse=True):
    #                     if isinstance(self.local_P[node][0], float):
    #                         del self.local_P[node][index:index+2]
    #                     else:
    #                         del self.local_P[node][index]
    #                 if not self.local_P[node]:
    #                     self.local_P[node].append(-node.cost())#,node_children.f1(model_zero,y_train,i)])
    #                     self.local_P[node].append(node.f1(model_zero,y_train,i))

    #     else:
    #         self.global_P = [list(x) for x in set(tuple(x) for x in self.global_P)]
    #         for node in path:
    #             local = []
    #             if isinstance(self.local_P[node][0],float):
    #                 local.append([i/(self.N[node] + 1) for i in self.local_P[node]])
    #             else: #list of lists
    #                 for j in range(len(self.local_P[node])):
    #                     local.append([i/(self.N[node] + 1) for i in self.local_P[node][j]])

    #             n_dom = []
    #             for k in range(len(local)):
    #                 for z in range(len(self.global_P)):
    #                     n_dom.append(dominates(local[k],self.global_P[z]))

    #             n_dom = list(set(n_dom))

    #             true_indx = [i for i, x in enumerate(n_dom) if x] #index of list of lists

    #             true_indx = list(set(true_indx))

    #             if true_indx:
    #                 for index in sorted(true_indx, reverse=True):
    #                     if isinstance(self.local_P[node][0], float):
    #                         del self.local_P[node][index:index+2]
    #                     else:
    #                         del self.local_P[node][index]
    #                 if not self.local_P[node]:
    #                     self.local_P[node].append(-node.cost())#,node_children.f1(model_zero,y_train,i)])
    #                     self.local_P[node].append(node.f1(model_zero,y_train,i))

    def _uct_select(self, node):

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node]) #n = child

        log_N_vertex = math.log(self.N[node])

        N_vertex = self.N[node]

        def uct(n):
            r_norm = []
            r_norm_lst = []
            for i in range(len(self.local_P[n])): #could be list of lists
                r_lst = []
                if isinstance(self.local_P[n][0],float): #this is when only one list 
                    #if self.local_P[n][1] < 1.0:
                    #print(self.local_P[n][i] / N_vertex)
                    #print(self.exploration_weight * math.sqrt(log_N_vertex / N_vertex)) #really small compared to the above
                    r_norm.append(self.local_P[n][i] / N_vertex + self.exploration_weight * math.sqrt(log_N_vertex / N_vertex))
                    #else:
                    #r_norm.append(self.local_P[n][i] + self.exploration_weight * math.sqrt(log_N_vertex / self.N[n]))
                else: #list of lists
                    #if self.local_P[n][i][1] < 1.0:
                    #    for j in range(len(self.local_P[n][i])):
                    #        r_lst.append(self.local_P[n][i][j] + self.exploration_weight * math.sqrt(log_N_vertex / self.N[n]))
                    #else:
                    for j in range(len(self.local_P[n][i])):
                        r_lst.append(self.local_P[n][i][j] / N_vertex + self.exploration_weight * math.sqrt(log_N_vertex / N_vertex))
                r_norm_lst.append(r_lst)

            #print(r_norm)
            #print(r_norm_lst)

            if isinstance(self.local_P[n][0],float):
                return hypervolume(r_norm)
            else:
                return hypervolume_lst(r_norm_lst)

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