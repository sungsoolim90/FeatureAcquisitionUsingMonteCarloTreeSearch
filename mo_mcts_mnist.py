'''
This script implements the multiobjective (MO) Monte Carlo Tree Search algorithm
for the feature acquisition problem of MNIST. 

a) Each 4x4 block of pixels is defined as a feature with cost of 16
b) Feature acquisition episode starts from the empty feature state and ends in the full feature state

parameters_MCTS.py file contains the hyperparameters in the MO-MCTS algorithm. 

outputs:

1) Constructed tree search at a specified save frequency
2) Trained policy network weights
2) Inference samples

'''

from multiprocessing import Pool
from monte_carlo_tree_search_mo import MCTS, Node
from tree_mo import Tree

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import time, pickle, json, random, heapq
import collections, functools, operator 

from parameters_MCTS import*

# Set a seed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)# 2. Seed python built-in pseudo-random generator
np.random.RandomState(seed_value)# 3. Seed numpy pseudo-random generator

# Setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0

x = np.concatenate((x_train,x_test))
y = np.concatenate((y_train,y_test))

# Split based on input random seeds
train_size = 0.8

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=train_size,random_state=random_state)

x_train = x_train.reshape((-1,28,28,1))
x_test = x_test.reshape((-1,28,28,1))

# Saved pretrained classifiers
filename = 'mnist_' + classifier_name + '_' + model_name + '_' + str(random_state) + '_' + str(seed_value)

if classifier_name == 'lr':

    weight_name = filename + '.h5'

    model_class = pickle.load(open(weight_name, 'rb'))

else:

    weight_name = filename + '.h5'
    opt_name = filename + '_optimizer.pkl'

    model_class = Sequential()

    model_class.add(Conv2D(filters=64, kernel_size = (3,3), dilation_rate = (2,2), padding = 'same', activation="relu", input_shape=(28,28,1)))
    model_class.add(MaxPooling2D(pool_size=(2,2)))

    model_class.add(Conv2D(filters=128, kernel_size = (3,3), dilation_rate = (2,2), padding = 'same', activation="relu"))
    model_class.add(MaxPooling2D(pool_size=(2,2)))

    model_class.add(Conv2D(filters=256, kernel_size = (3,3), dilation_rate = (2,2), padding = 'same', activation="relu"))
    model_class.add(MaxPooling2D(pool_size=(2,2)))
        
    model_class.add(Flatten())

    model_class.add(Dense(512,activation="relu"))
        
    model_class.add(Dense(10,activation="softmax"))

    # # # Instantiate an optimizer.
    lr = 1e-6
    epochs = 101

    with open(opt_name, 'rb') as f:
        weight_values = pickle.load(f)

    grad_vars = model_class.trainable_weights

    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

    zero_grads = [tf.zeros_like(w) for w in grad_vars]

    # Apply gradients which don't do nothing with Adam
    optimizer.apply_gradients(zip(zero_grads, grad_vars))

    # Set the weights of the optimizer
    optimizer.set_weights(weight_values)

    # Set the trainable weights of the model
    model_class.load_weights(weight_name)

    # # Needed for retrain strategy where CNN or LR are periodically retrained
    # # Instantiate a loss function.
    # loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    # # Prepare the metrics.
    # train_acc_metric = keras.metrics.CategoricalAccuracy()
    # val_acc_metric = keras.metrics.CategoricalAccuracy()

    # @tf.function
    # def train_step(x, y):
    #     with tf.GradientTape() as tape:
    #         logits = model_class(x, training=True)
    #         loss_value = loss_fn(y, logits)
    #     grads = tape.gradient(loss_value, model_class.trainable_weights)
    #     optimizer.apply_gradients(zip(grads, model_class.trainable_weights))
    #     train_acc_metric.update_state(y, logits)
    #     return loss_value

    # @tf.function
    # def test_step(x, y):
    #     val_logits = model_class(x, training=False)
    #     loss_value = loss_fn(y,val_logits)
    #     val_acc_metric.update_state(y, val_logits)
    #     return loss_value

    # def train(train_dataset,val_dataset):
    #     loss_train = []
    #     acc_train = []
    #     loss_test = []
    #     acc_test = []
    #     for epoch in range(epochs):
    #         print("\nStart of epoch %d" % (epoch,))
    #         start_time = time.time()
    #         loss_train_epoch = 0.0
    #         # Iterate over the batches of the dataset.
    #         for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
    #             loss_value = train_step(x_batch_train, y_batch_train)
    #             loss_train_epoch += loss_value
    #         loss_train_epoch /= (step+1)
    #         loss_train.append(loss_train_epoch)

    #         # Display metrics at the end of each epoch.
    #         train_acc = train_acc_metric.result()
    #         acc_train.append(train_acc)

    #         print("Training acc over epoch: %.4f" % (float(train_acc),))
    #         print("Training loss over epoch: %.4f" % (float(loss_train_epoch),))

    #         # Reset training metrics at the end of each epoch
    #         train_acc_metric.reset_states()

    #         loss_test_epoch = 0.0
    #         # Run a validation loop at the end of each epoch.
    #         for step_val, (x_batch_val, y_batch_val) in enumerate(val_dataset):
    #             loss_value_test = test_step(x_batch_val, y_batch_val)
    #             loss_test_epoch += loss_value_test
    #         loss_test_epoch /= (step_val+1)
    #         loss_test.append(loss_test_epoch)

    #         val_acc = val_acc_metric.result()

    #         acc_test.append(val_acc)

    #         print("Validation acc: %.4f" % (float(val_acc),))
    #         print("Validation loss: %.4f" % (float(loss_test_epoch),))

    #         val_acc_metric.reset_states()

    #         print("Time taken: %.2fs" % (time.time() - start_time))

    #     return loss_train, loss_test, acc_train, acc_test

def hypervolume(front):
    reference = [-1.0, 0.0]
    #front = sorted(front, key=lambda x: x[0]) #sort by first objective (cost)
    hv = 0
    #for i in range(len(front)):
    h = front[1] - reference[1] #accuracy 
    #	if i == 0:
    hv += (front[0] - reference[0])*h
    #	else:
    #		hv += (front[i][0]/41.0 - front[i-1][0]/41.0)*h
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

def score(n):
    if tree.N[n] == 0:
        return float("inf")  #avoid unseen moves
    P = [x for x in tree.global_P if x]
    if isinstance(tree.local_P[n][0],float):
        local = [i/(tree.N[n] + 1) for i in tree.local_P[n]]
    else: #list of lists
        local = []
        for j in range(len(tree.local_P[n])):
            local.append([i/(tree.N[n] + 1) for i in tree.local_P[n][j]])

    P = [tree.global_P, local]
    P = flatten(P)
    row = int(len(P)/2.0)
    col = 2
    P = [P[col*i : col*(i+1)] for i in range(row)]

    P,dominated = simple_cull(P,dominates)

    P = list(P)
    
    for i in range(len(P)):
        P[i] = list(P[i])

    score = hypervolume_lst(P)

    return score

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

tf.random.set_seed(seed_value)

model = Sequential()

dilation_rate = 2

model.add(Conv2D(filters=64, kernel_size = (3,3), dilation_rate = dilation_rate, padding = 'same', activation = 'relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size = (3,3), dilation_rate = dilation_rate, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=256, kernel_size = (3,3), dilation_rate = dilation_rate, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
    
model.add(Flatten())

model.add(Dense(512,activation="relu"))
    
model.add(Dense(49,activation="softmax"))

# Instantiate an optimizer.
lr = 1e-5
epochs = 101

optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

# Instantiate a loss function.
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

# Prepare the metrics.
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        #logits = tf.reshape(logits,(-1,10))
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value

@tf.function
def val_step(x, y):
    val_logits = model(x, training=False)
    #val_logits = tf.reshape(val_logits,(-1,10))
    loss_value = loss_fn(y,val_logits)
    val_acc_metric.update_state(y, val_logits)
    return loss_value

def train(train_dataset,val_dataset):
    loss_train = []
    acc_train = []
    loss_test = []
    acc_test = []
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        loss_train_epoch = 0.0
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train)
            loss_train_epoch += loss_value
        loss_train_epoch /= (step+1)
        loss_train.append(loss_train_epoch)

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        acc_train.append(train_acc)

        print("Training acc over epoch: %.4f" % (float(train_acc),))
        print("Training loss over epoch: %.4f" % (float(loss_train_epoch),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        loss_test_epoch = 0.0
        # Run a validation loop at the end of each epoch.
        for step_val, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            loss_value_test = val_step(x_batch_val, y_batch_val)
            loss_test_epoch += loss_value_test
        loss_test_epoch /= (step_val+1)
        loss_test.append(loss_test_epoch)

        val_acc = val_acc_metric.result()

        acc_test.append(val_acc)

        print("Validation acc: %.4f" % (float(val_acc),))
        print("Validation loss: %.4f" % (float(loss_test_epoch),))

        val_acc_metric.reset_states()

        print("Time taken: %.2fs" % (time.time() - start_time))

    return loss_train, loss_test, acc_train, acc_test

model.compile(optimizer=optimizer, loss=loss_fn, metrics = [train_acc_metric])

def make_keys(result_iterable):

    keys_iterable, values = result_iterable

    global result
    global result_N

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

    def list_of_features(node,i,X_train):

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

        return to_be_acquired#.sort()

    def dominates(row, candidateRow):
        return sum([row[x] > candidateRow[x] for x in range(len(row))]) == len(row) 

    def hypervolume(front):
        reference = [-1.0, 0.0]
        #front = sorted(front, key=lambda x: x[0]) #sort by first objective (cost)
        hv = 0
        #for i in range(len(front)):
        h = front[1] - reference[1] #accuracy 
            #if i == 0:
        hv += (front[0] - reference[0])*h
            #else:
                #hv += (front[i][0]/41.0 - front[i-1][0]/41.0)*h
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

    def flatten(x):
        if isinstance(x, collections.Iterable):
            return [a for i in x for a in flatten(i)]
        else:
            return [x]

    def action_probs(node,all_X,scores_total,i,X_train):

        def score(n):

            if scores_total.get(n) and all_X.get(n):
                
                P = all_X.get(n)

                local_P = scores_total.get(n)

                P = [x for x in P if x]

                P = [P, local_P]
                P = flatten(P)
                row = int(len(P)/2.0)
                col = 2
                P = [P[col*i : col*(i+1)] for i in range(row)]
                score = hypervolume_lst(P)

                return score
            else:
                return float("0")  # avoid unseen

        acquired = list_of_features(node,i,X_train)
        
        total = [i for i in range(49)]

        scores = []
        for z in range(49):
            score_to_append = 0.0
            if total[z] in acquired:
                scores.append(0.0) #this hold all the time 
            else:
                score_to_append += score(node) #immediate reward (node)
                child = make(node,total[z],i,X_train)
                score_to_append += score(child) #first child
                scores.append(score_to_append)

        sum_scores = sum(scores)

        if sum_scores == 0.0:
            scores_return = scores
        else:
            scores_return = [e/sum_scores for e in scores]

        return scores_return

    keys = []
    scores = []
    indx_done = []

    key = keys_iterable.tup

    print(key)

    value_iterable = values

    sum_lst = []
    for z in range(len(x_train_new_linear)):
        sum_lst.append(sum([x == y for (x,y) in zip(list(x_train_new_linear.iloc[z,:]),key)]))
    
    if np.mean(sum_lst) > 10:
        indx = np.argwhere(sum_lst == np.amax(sum_lst)).flatten().tolist()
        indx_copy = indx.copy()
        indx_done_copy = indx_done.copy()
        while indx:
            min_indx = np.min(indx)
            if min_indx not in indx_done:
                indx_done.append(min_indx)
                break
            else:
                indx_to_be_appended = np.argwhere(indx==min_indx).flatten().tolist() #can be multiple
                indx.pop(indx_to_be_appended[0])
        if not indx and len(indx_done) == len(indx_done_copy):
            while indx_copy:
                min_indx_copy = np.min(indx_copy)
                count = indx_done.count(min_indx_copy)
                if count <= 100:
                    indx_done.append(min_indx_copy)
                    break
                else:
                    indx_copy_to_be_appended = np.argwhere(indx_copy == min_indx_copy).flatten().tolist()
                    indx_copy.pop(indx_copy_to_be_appended[0])
    else:
        indx = []

    if indx_done:
        keys.append(key)
        scores.append(action_probs(keys_iterable,result_N,result,indx_done[-1],x_train_new))

    print(scores)
    return keys, scores

def flatten(x):
    if isinstance(x, collectionsAbc.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def dominates(row, candidateRow):
    return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row) 

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
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints

#Training loop
m,n,o,p = x_train.shape

node_total_total = []
tree_q_total = []
tree_N_total = []

for v in range(num_epochs):
	node_total = []
	hv_total = []
	hv_hv_total = []
	P_total = []
	tree_q = []
	tree_N = []

	node_total_retrain = []
	X_train_z_copy = x_train.copy()
	y_train_copy = y_train.copy()
	retrain_time = 1
	purge_time = 1

	for i in range(m):
		#start with a patient with zero features
		print(i)
		tp = (1.0e-6,)*784
		node = Tree(tup=tp, terminal=False) #the model is called multiple times
		#num = np.random.randint(1,15)
		#num_node = 1
		tree = MCTS() #for every new sample
		#tree.update(node,model_zero)
		hv = []
		hv_hv = []
		act = []
		nodes = []
		while True:
			#based on the given node, find the MCTS next state
			hv_node = [-node.cost(i,x_train_new),node.f1(model_zero,y_train_new,i,x_train_new)]

			nodes.append(node.tup)

			if node.terminal:
				if isinstance(tree.local_P[node][0],float):
					hv_node_save = [i / (tree.N[node]+1) for i in tree.local_P[node]]
				else:
					hv_node_save_lst = []
					for j in range(len(tree.local_P[node])):
						hv_node_save_lst.append([i / (tree.N[node]+1) for i in tree.local_P[node][j]])
					hv_node_save = flatten(hv_node_save_lst)
					row = int(len(hv_node_save)/2.0)
					col = 2
					hv_node_save = [hv_node_save[col*i : col*(i+1)] for i in range(row)]

				hv_hv.append(hypervolume(hv_node))
				hv.append(hv_node)
				break

			for z in range(training):
				print(z) 
				tree.train(node,i,model_zero,x_train,y_train)

			if integrated:

				tup_action = node.tup
				tup_action = np.array(tup_action).reshape((-1,28,28,1))
				logits = model(tup_action,training=False)
				logits = np.array(logits)
				logits = logits[0]

				indx = np.argwhere(logits == np.amax(logits)).flatten().tolist()

				if len(indx) == 49:
					action = random.choice(indx)
					while action in act:
						action = random.choice(indx)
					act.append(action)
				else:
					action = heapq.nlargest(49, range(len(logits)), key=logits.__getitem__)
					st =  set(act)
					to_be = [ele for ele in action if ele not in st]
					action = to_be[0]
					act.append(action)

				node = make(node,action,i,x_train_new)
			else:
				node, score_node = tree.choose(node,i,x_train_new) #choose best score next state

			if isinstance(tree.local_P[node][0],float):
				hv_node_save = [i / (tree.N[node]+1) for i in tree.local_P[node]]
			else:
				hv_node_save_lst = []
				for j in range(len(tree.local_P[node])):
					hv_node_save_lst.append([i / (tree.N[node]+1) for i in tree.local_P[node][j]])
				hv_node_save = flatten(hv_node_save_lst)
				row = int(len(hv_node_save)/2.0)
				col = 2
				hv_node_save = [hv_node_save[col*i : col*(i+1)] for i in range(row)]

			dict_node = {}
			dict_node[node] = hv_node_save
			tree_q.append(dict_node)

			hv_hv.append(hypervolume(hv_node))
			hv.append(hv_node)

			P = [x for x in tree.global_P if x]
			P_node = {}
			P_node[node] = P
			tree_N.append(P_node)

		hv_total.append(hv)
		hv_hv_total.append(hv_hv)

		node_total.append(nodes)
		P = [x for x in tree.global_P if x]

		if isinstance(P[0],float):
			P_total.append(P)
		else:
			P = [list(x) for x in set(tuple(x) for x in P)]
			P_total.append(P)

		if integrated: 

			if (i+1) % retrain_step == 0:

				print('Training')

				result = {}
				for k in set().union(*tree_q):
					for d in tree_q:
						if d.get(k):
							result[k] = d.get(k)
				
				result_N = {}
				for k in set().union(*tree_N):
					for d in tree_N:
						if d.get(k):
							result_N[k] = d.get(k)

				p = Pool(8)
				finalValue = list(p.map(make_keys, result.items()))

				keys = []
				scores = []
				for k in range(len(finalValue)):
					if finalValue[k][0]:
						keys.append(list(finalValue[k][0][0]))
						scores.append(finalValue[k][1][0])

				all_X_filter = [] 
				action = []
				for t in range(len(scores)):
					criterion = list(scores[t])
					common_indx = np.argwhere(criterion==np.max(criterion)).flatten().tolist()#np.argmax(criterion)
					if len(common_indx) == 1:
						action.append(scores[t])#random.choice(common_indx))
						all_X_filter.append(keys[t])

				all_X_filter = np.array(all_X_filter)
				action = np.array(action)

				finalValue = []
				keys = []
				scores = []

				train_X_total, test_X_total, train_y_total, test_y_total = train_test_split(all_X_filter,action,test_size = 0.2,random_state=random_state)

				batch_size = 2
				train_X_total = np.reshape(train_X_total, (-1, 28,28,1))
				test_X_total = np.reshape(test_X_total, (-1, 28,28,1))
				train_dataset = tf.data.Dataset.from_tensor_slices((train_X_total, train_y_total))
				train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

				val_dataset = tf.data.Dataset.from_tensor_slices((test_X_total, test_y_total))
				val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)

				tf.config.run_functions_eagerly(True)

				loss_train, loss_test, acc_train, acc_test = train(train_dataset,val_dataset)#$#keras_models[m])

				weight_name = 'mnist_mo_lr_model_weights_' + model_name + '_' + save_name + '_' + further_name + '_' + str(random_state) + '_' + str(seed_value) + '.h5'
				model.save_weights(weight_name)

				tree_q = []
				tree_N = []

		if retrain_sk:

			if (i + 1) % retrain_step == 0:

				print('Retraining CNN')

				if purge_time > 1:
					y_train_new_z = []
					for k in range(purge_step*(purge_time-1),i+1):
						y_train_new_z.append(y_train_copy[k])
				else:
					y_train_new_z = []
					for k in range(i+1):
						y_train_new_z.append(y_train_copy[k])

				lst_to_be_added = list(set([i for i in range(10)]) - set(y_train_new_z))
				indx_to_be_added = []
				if lst_to_be_added:
					for w in range(len(lst_to_be_added)):
						l = np.where(y_train_new == lst_to_be_added[w])
						indx_to_be_added.append(l[0][0])
						y_train_new_z.append(lst_to_be_added[w])

				new_nodes = []

				for b in range(len(indx_to_be_added)):
				    new_nodes.append(x_train_new[indx_to_be_added[b]].reshape((-1,784))[0])

				y_train_new_z_total = pd.DataFrame(y_train_new_z)

				y_train_new_z_total = pd.concat([pd.DataFrame(y_train_copy),y_train_new_z_total],ignore_index=True)

				retrain_node_transform_selected = pd.DataFrame(node_total_retrain)

				X_train_z_new = pd.concat([pd.DataFrame(X_train_z_copy.reshape((-1,784))),retrain_node_transform_selected,pd.DataFrame(new_nodes)],ignore_index = True)
				
				model_zero.fit(X_train_z_new, y_train_new_z_total)
				# model_zero.fit(node_retrain_total,y_train_new)

				#batch_size = 512
				#train_X_total = np.reshape(X_train_z_new, (-1, 27))
				#test_X_total = np.reshape(X_test_z, (-1, 27))
				#train_dataset = tf.data.Dataset.from_tensor_slices((train_X_total, y_train_new))
				#train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

				#val_dataset = tf.data.Dataset.from_tensor_slices((test_X_total, y_test))
				#val_dataset = val_dataset.batch(batch_size)

				#loss_train, loss_test, acc_train, acc_test = train_sk(train_dataset,val_dataset)

			if (i + 1) % purge_step == 0:

				print('Purging')

				q,r = X_train_z_new.shape
				m,n,o,p = x_train_new.shape
				purge = q - m

				indx = random.sample(range(0, q), purge)

				indx_name = []

				for j in range(len(indx)):
					indx_name.append(X_train_z_new.iloc[indx[j],:].name)

				X_train_drop = X_train_z_new.drop(index=indx_name)

				y_train_drop = y_train_new_z_total.drop(index=indx_name)

				node_total_retrain = []

				X_train_z_copy = X_train_drop.to_numpy()

				y_train_copy = y_train_drop.iloc[:,0].tolist()

				purge_time += 1

	if (v + 1) % num_epochs == 0:

		print('Saving tree')

		result = {}
		for k in set().union(*tree_q):
			for d in tree_q:
				if d.get(k):
					result[k] = d.get(k)
        
		result_N = {}
		for k in set().union(*tree_N):
			for d in tree_N:
				if d.get(k):
					result_N[k] = d.get(k)

		filename_Q = 'tree_P_' + model_name + '_lr_' + further_name + '_' + save_name + str(coeff) + '_' + str(random_state) + '_' + str(seed_value) + '.pickle'
		with open(filename_Q, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

		filename_N = 'tree_NMO_' + model_name + '_lr_' + further_name + '_' + save_name +  str(coeff) + '_' + str(random_state) + '_' + str(seed_value) + '.pickle'
		with open(filename_N, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(result_N, f, pickle.HIGHEST_PROTOCOL)

		tree_q = []
		tree_N = []

if retrain_sk:
	filename = 'retrained_mo_hf_smote_lr_' + model_name + '_' + further_name + '_' + save_name + '_' +  str(coeff) + '_' + str(random_state) + '_' + str(seed_value) + '_' + str(retrain_step) + '.h5'
	#model_zero.save_weights(filename)
	pickle.dump(model_zero, open(filename, 'wb'))