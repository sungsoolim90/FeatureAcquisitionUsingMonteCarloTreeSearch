from multiprocessing import Pool
from multiprocess import Pool as mp
from monte_carlo_tree_search import MCTS, Node
from tree import Tree

from sklearn.linear_model import LogisticRegression as sk
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import time
import collections, functools, operator 
import pickle
import json
import random
import heapq


from parameters_MCTS import*


#Set a seed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.RandomState(seed_value)

#Setup train and test splits
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0

x = np.concatenate((x_train,x_test))
y = np.concatenate((y_train,y_test))

train_size = 0.8

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=train_size,random_state=random_state)

x_train = x_train.reshape((-1,28,28,1))
x_test = x_test.reshape((-1,28,28,1))

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


# import pickle

# if pretrain_model or fit_model:
# 	filename = 'finalized_model_mnist_' + str(random_state) + '_' + str(seed_value) + '.sav'
# 	#filename = 'physionet' + '_' + str(random_state) + '_' + str(seed_value)# + '_quadratic_end'
# 	#weight_name = filename + '.h5'

# 	# # # NOW set the trainable weights of the model
# 	#model_sk.load_weights(weight_name)

# elif retrain_model:
# 	filename = 'finalized_model_mnist_' + str(random_state) + '_' + str(seed_value) + '.sav'

# 	#filename = 'physionet' + '_' + str(random_state) + '_' + str(seed_value)# + '_quadratic_end'
	
# 	#weight_name = filename + '.h5'

# 	#opt_name = filename + '_optimizer.pkl'

# 	#with open(opt_name, 'rb') as f:
# 	#	weight_values = pickle.load(f)

# 	#grad_vars = model_zero.trainable_weights

# 	#optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

# 	#zero_grads = [tf.zeros_like(w) for w in grad_vars]

# 	#Apply gradients which don't do nothing with Adam
# 	#optimizer.apply_gradients(zip(zero_grads, grad_vars))

# 	#Set the weights of the optimizer
# 	#optimizer.set_weights(weight_values)

# elif random_model:
# 	if beg or end:
# 		#filename = 'physionet_random' + '_' + str(coeff) + '_' + str(random_state) + '_' + str(seed_value) + '_' + cost_name + '_' + further_name# + '.h5'#quadratic_end'
# 		#filename = 'physionet_random' + '_' + str(random_state) + '_' + str(seed_value)# + '_' + cost_name + '_' + further_name# + '.h5'#quadratic_end'
# 		#filename = 'finalized_model_physionet_balanced_random_' + str(random_state) + '_' + str(seed_value) + '.sav'
# 		filename = 'finalized_model_mnist_smote_' + model_name + '_' + str(random_state) + '_' + str(coeff) + '_' + str(seed_value) + '_' +  cost_name + '_' + further_name + '.sav'
# 		#weight_name = filename + '.h5'
# 		#model_zero.load_weights(weight_name)
# 	#else:
# 	#	filename = 'finalized_model_physionet_smote_' + str(random_state) + '_' + str(seed_value) + '.sav'

# #model_sk.load_weights(weight_name)

# model_sk = pickle.load(open(filename, 'rb'))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.regularizers import l2
#from tensorflow.keras.regularizers import l1_l2

from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__

# Run the function
make_keras_picklable()

# #import tensorflow as tf
# #from tensorflow import keras
# #filename = 'hf_random_' + str(random_state) + '_' + str(random_state) + '_.h5'
# #model_sk = keras.models.load_model(filename)#, custom_objects={'Functional':keras.models.Model})
filename = 'mnist_cnn_random' + '_' + str(random_state) + '_' + str(seed_value)# + '_quadratic_end'
weight_name = filename + '.h5'
opt_name = filename + '_optimizer.pkl'

model_sk = Sequential()

dilation_rate = 2

#model.add(Lambda(standardize,input_shape=(28,28,1)))    
model_sk.add(Conv2D(filters=64, kernel_size = (3,3), dilation_rate = dilation_rate, padding = 'same', activation = 'relu', input_shape=(28,28,1)))
#model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))
model_sk.add(MaxPooling2D(pool_size=(2,2)))
#model.add(BatchNormalization())

model_sk.add(Conv2D(filters=128, kernel_size = (3,3), dilation_rate = dilation_rate, padding = 'same', activation = 'relu'))
#model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
model_sk.add(MaxPooling2D(pool_size=(2,2)))
#model.add(BatchNormalization())    

model_sk.add(Conv2D(filters=256, kernel_size = (3,3), dilation_rate = dilation_rate, padding = 'same', activation = 'relu'))
model_sk.add(MaxPooling2D(pool_size=(2,2)))
#model.add(BatchNormalization())
    
model_sk.add(Flatten())

model_sk.add(Dense(512,activation="relu"))
    
model_sk.add(Dense(10,activation="softmax"))

# Instantiate an optimizer.
lr = 1e-5
epochs = 101
#decay_rate = lr/epochs

with open(opt_name, 'rb') as f:
    weight_values = pickle.load(f)

grad_vars = model_sk.trainable_weights

optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

# Instantiate a loss function.
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

zero_grads = [tf.zeros_like(w) for w in grad_vars]

#Apply gradients which don't do nothing with Adam
optimizer.apply_gradients(zip(zero_grads, grad_vars))

#Set the weights of the optimizer
optimizer.set_weights(weight_values)

# # # NOW set the trainable weights of the model
model_sk.load_weights(weight_name)

@tf.function
def train_sk_step(x, y):
    with tf.GradientTape() as tape:
        logits = model_sk(x, training=True)
        #logits = tf.reshape(logits,(-1,10))
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model_sk.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_sk.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value

@tf.function
def val_sk_step(x, y):
    val_logits = model_sk(x, training=False)
    #val_logits = tf.reshape(val_logits,(-1,10))
    loss_value = loss_fn(y,val_logits)
    val_acc_metric.update_state(y, val_logits)
    return loss_value

def train_sk(train_dataset,val_dataset):
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
            loss_value = train_sk_step(x_batch_train, y_batch_train)
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
            loss_value_test = val_sk_step(x_batch_val, y_batch_val)
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

if integrated: 

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from tensorflow.keras.models import Sequential
    import tensorflow.keras.backend as K
    from tensorflow.keras.optimizers import Adam

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
    epochs = 501

    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

    # Instantiate a loss function.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # Prepare the metrics.
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

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
        
        def action_probs(node,all_X,scores_total,i,X_train):

            def score(n):
                if scores_total.get(n) and all_X.get(n):
                    return float(all_X.get(n) / scores_total.get(n))  # average reward
                else:
                    return float("0")  # avoid unse

            acquired = list_of_features(node,i,X_train)
            
            total = [i for i in range(49)]

            #feature_values = [value for k, value in enumerate(node.tup) if value != -1]

            #sum_lst = []
            #for z in range(len(X_train_z)):
            #    sum_lst.append(sum([x == y for (x,y) in zip(list(X_train_z.iloc[z,:]),feature_values)]))

            #indices = [k for k in range(len(sum_lst)) if sum_lst[k] > len(list_of_features(node.tup))] #indices of all children in the dictionary

            ##need to find the relationships

            scores = []
            for z in range(49):
                score_to_append = 0.0
                if total[z] in acquired:
                    scores.append(0.0) #this hold all the time 
                else:
                    score_to_append += score(node) #immediate reward (node)
                    child = make(node,total[z],i,X_train)
                    score_to_append += score(child) #first child
                    #child_score = score(child)
                    #while child_score != 0.0:
                    #    child = make_child(node,total[j],i)
                    #    score_to_append += score(child)
                    #    child_score = score(child)
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
        #i = 0
        #j = 0

        key = keys_iterable.tup

        print(key)

        value_iterable = values#[i]

        #feature_values = [value for i, value in enumerate(key) if value != -1] #based on real-valued features?
        #if feature_values:
        #need to keep the position
        sum_lst = []
        for z in range(len(x_train_new_linear)):
            sum_lst.append(sum([x == y for (x,y) in zip(list(x_train_new_linear.iloc[z,:]),key)]))#feature_values)]))
        
        #print(sum_lst)
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

m,n,o,p = x_train_new.shape

from multiprocessing import cpu_count

def one_iter(node,i,model_sk,X_train,act):

	Q = defaultdict(int)  # total reward of each node
	N = defaultdict(int)  # total visit count for each node
	children = dict()  # children of each node
	exploration_weight = 1.0 #constant for UCB
	act = []

	def _uct_select(node,Q,N,children,exploration_weight):

	    # All children of node should already be expanded:
	    assert all(n in children for n in children[node]) #n = child

	    log_N_vertex = math.log(N[node])

	    N_vertex = N[node]

	    def uct(n,exploration_weight):
	        return Q[n] / N_vertex + exploration_weight * math.sqrt(log_N_vertex / N_vertex)

	    return max(children[node], key=uct)

	def _select(node,Q,N,children,exploration_weight):
	    path = []
	    while True:

	        path.append(node)

	        if node not in children or not children[node]:
	            return path

	        #there is no unexplored
	        unexplored = children[node] - children.keys()

	        if unexplored:
	            n = unexplored.pop()
	            path.append(n)
	            return path

	        node = _uct_select(node,Q,N,children,exploration_weight)  # descend a layer deeper

	def _expand(node,i,X_train,Q,N,children,act):
	    
	    if node in children:
	        return  # already expanded
	    if node.terminal:
	        return
	    else:
	        children[node] = node.find_children(i,X_train,act)

	def _simulate(node,i,model_zero,X_train,act):
	    cum_reward = 0.0
	    z = 0
	    action_lst = act.copy()
	    while True:

	        print(z)
	        z += 1

	        cum_reward += node.reward(model_zero,X_train,i,action_lst)
	        
	        if node.is_terminal():
	            break

	        node, action = node.find_random_child(i,X_train,action_lst)

	        action_lst.append(action)

	    return cum_reward

	def _backpropagate(path,reward,Q,N,children):
	    
	    for node in reversed(path):
	        N[node] += 1
	        Q[node] += reward

	def train(node,i,model_zero,X_train,Q,N,children,exploration_weight,act):
	    print('select')
	    path = _select(node,Q,N,children,exploration_weight)
	    leaf = path[-1] #leaf node
	    print('expand')
	    _expand(leaf,i,X_train,Q,N,children,act) #children according to transition probability function - from the leaf node, expand
	    #print(children)
	    print('simulate')
	    cum_reward = _simulate(leaf,i,model_zero,X_train,act) #simulate to the end (random children)
	    #print(cum_reward)
	    print('backpropagate')
	    _backpropagate(path,cum_reward,Q,N,children) #backpropagate to the root (increase reward and N for each node by 1)

	train(node,i,model_sk,x_train_new,Q,N,children,exploration_weight,act)

	#print('one iter')

	return Q, N#, children

def wrap(args):
	return one_iter(*args)

def choose_next_child(node,i,X_train,Q,N,children,act): #Choose the node with highest score
    
    if node.is_terminal():
        raise RuntimeError(f"choose called on terminal node {node}")

    def score(n):
        if N[n] == 0:
            return float("-inf")  # avoid unseen moves
        return Q[n] / N[n]  # average reward

    if node not in children:
    	max_child = node.find_random_child(i,X_train,act)
    else:
        max_child = max(children[node],key=score)
    
    return max_child

node_total_total = []
#save_step = m#512
purge_step = int(m/2.0)
save_step = purge_step

from collections import defaultdict

#retrain_step = 6000
for j in range(1):
	node_total = []
	node_total_retrain = []
	act_total = []
	#np.random.seed()
	#training = 100
	tree_q = []
	tree_N = []
	X_train_z_copy = x_train_new.copy()
	y_train_copy = y_train_new.copy()
	retrain_time = 1
	purge_time = 1
	#tree = MCTS()
	#act = []
	for i in range(10):#m):
		#start with a patient with zero features
		print(i)
		tp = (1.0e-6,)*784
		node = Tree(tup=tp, terminal=False) #the model is called multiple times
		tree = MCTS() #for every new sample
		act  = []
		#vt = 1
		while True:#num_node<num:

			for z in range(training):
				print(z)
				tree.train(node,i,model_sk,x_train_new)

			if node.terminal:

				print('done')
				break

			# p = Pool(16)
			# argStorage = [(node,i,model_sk,x_train_new,act)] * training
			# f_val = list(p.map(wrap,argStorage))
			# p.close()
			# p.join()

			# tree_q_step = []
			# tree_N_step = []
			# children_step = []
			# for j in range(len(f_val)):
			# 	tree_q_step.append(f_val[j][0])
			# 	tree_N_step.append(f_val[j][1])
			# 	children_step.append(f_val[j][2])

			# f_val = []

			# #tree.train(node,i,model_sk,x_train_new)

			# print('Saving')
			# result = {} 
			# for d in tree_q_step: 
			# 	for k in d.keys(): 
			# 		result[k] = result.get(k, 0) + d[k]

			# tree_q_step =[]
			
			# result_N = {} 
			# for d in tree_N_step: 
			# 	for k in d.keys(): 
			# 		result_N[k] = result_N.get(k, 0) + d[k]

			# tree_N_step = []

			# for z in range(training):
			# 	print(z) 
			# 	tree_training = MCTS()
			# 	tree_training.train(node,i,model_sk,x_train_new)

			if integrated:

				tup_action = node.tup
				tup_action = np.array(tup_action).reshape((-1,28,28,1))#size)
				logits = model(tup_action,training=False)
				logits = np.array(logits)
				logits = logits[0]

				indx = np.argwhere(logits == np.amax(logits)).flatten().tolist()

				if len(indx) == 49:
                    #st = set(act)
                    #while 
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
				#this is the only place where act becomes one element longer 
				node = tree.choose(node,i,x_train_new)
				#act.append(action)

			act_acquired = list_of_features(node,i,x_train_new)

			print(act_acquired)
			
			tree_q.append(tree.Q)#result)
			tree_N.append(tree.N)#result_N)
			#act_total.append(act)

		if retrain_model:

			tp = (1.0e-6,)*784
        #    state = np.array(tp,dtype=np.float64)		
			node_retrain = Tree(tup=tp, terminal=False) #the model is called multiple times
		#num = np.random.randint(1,15)			node_retrain = Tree(tup=tup, terminal=False) #the model is called multiple times
			num = np.random.randint(1,50)
			num_node = 1
			act_retrain = []
			#tree_retrain = MCTS() #for every new sample

			while num_node<num:
			# 	#based on the given node, find the MCTS next state
				if node_retrain.terminal:
					break

				# p = Pool(8)
				# argStorage = [(node_retrain,i,model_sk,x_train_new)] * 50
				# f_val = list(p.map(wrap,argStorage))
				# p.close()
				# p.join()

				# tree_q_step = []
				# tree_N_step = []
				# for j in range(len(f_val)):
				# 	tree_q_step.append(f_val[j][0])
				# 	tree_N_step.append(f_val[j][1])

				# f_val = []

				#node_retrain = tree_retrain.choose(node_retrain,i,X_train_new) #choose best score next state

				node_retrain = node_retrain.find_random_child(i,x_train_new)

				#act_retrain.append(action)

				num_node += 1
			
			node_total_retrain.append(node_retrain.tup)

		#tree_q.append(tree.Q)
		#tree_N.append(tree.N)

		if (i+1) % save_step == 0:

			print('Saving')
			result = {} 
			for d in tree_q: 
				for k in d.keys(): 
					result[k] = result.get(k, 0) + d[k]
			
			result_N = {} 
			for d in tree_N: 
				for k in d.keys(): 
					result_N[k] = result_N.get(k, 0) + d[k]

			filename_Q = 'tree_Q_' + cost_name + '_' + model_name + '_' + further_name + '_cnn_' + save_name + '_' + str(coeff) + '_'  + str(random_state) + '_' + str(training) + '_1_' + str(retrain_step) + '_' + str(seed_value) + '_' + str(i+1) + '.pickle'
			with open(filename_Q, 'wb') as f:
				# Pickle the 'data' dictionary using the highest protocol available.
				pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

			filename_N = 'tree_N_' + cost_name + '_' + model_name + '_' + further_name + '_cnn_' + save_name + '_' + str(coeff) + '_'  + str(random_state) + '_' + str(training) + '_1_' + str(retrain_step) + '_' + str(seed_value) + '_' + str(i+1) + '.pickle'
			with open(filename_N, 'wb') as f:
				# Pickle the 'data' dictionary using the highest protocol available.
				pickle.dump(result_N, f, pickle.HIGHEST_PROTOCOL)

			tree_q = []
			tree_N = []
            
			if retrain_sk:
				filename = 'retrained_' + cost_name + '_' + further_name + '_model_mnist_cnn_' + save_name + '_' + str(coeff) + '_' + str(random_state) + '_' + str(training) + '_1_' + str(retrain_step) + '_' + str(seed_value) + '.h5'
				model_sk.save_weights(filename)
				#pickle.dump(model_sk, open(filename, 'wb'))

			# name = 'action_train_' + cost_name + '_' + model_name + '_' + further_name + '_cnn_physionet_mcts_' + save_name + '_' + str(coeff) + '_' + str(random_state) + '_' + str(training) + '_1_' + str(retrain_step) + '_' + str(seed_value) + '_' + str(j) + '.txt'
			# with open(name, 'w') as f:
			# 	f.write(json.dumps(act_total))

			act_total = []

		if integrated: 

			if (i+1) % retrain_step == 0:
				
				print('Training')

				result = {} 
				for d in tree_q: 
					for k in d.keys(): 
						result[k] = result.get(k, 0) + d[k] 

				result_N = {} 
				for d in tree_N: 
					for k in d.keys(): 
						result_N[k] = result_N.get(k, 0) + d[k]

				#keys,scores = make_keys(result,result_N)

				p = Pool(8)

				from itertools import islice

				def take(n, iterable):
					"Return first n items of the iterable as a list"
					return list(islice(iterable, n))

				#n = 4000

				#n_items = take(n, result.items())
				#n_items = dict(n_items)
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
					common_indx = np.argwhere(criterion==np.max(criterion)).flatten().tolist()
					if len(common_indx) == 1:#49:
						action.append(random.choice(common_indx))
						all_X_filter.append(keys[t])
				
				all_X_filter = np.array(all_X_filter)
				action = np.array(action)

				# name = 'all_X_filter_27_1_' + str(random_state) + '_' + str(seed_value) + '_' + str(i+1) + '_1.pkl'
				# with open(name, 'wb') as f:
				# 	pickle.dump(all_X_filter, f)

				# name = 'action_27_1_' + str(random_state) + '_' + str(seed_value) + '_' + str(i+1) + '_1.pkl'
				# with open(name, 'wb') as f:
				# 	pickle.dump(action, f)

				train_X_total, test_X_total, train_y_total, test_y_total = train_test_split(all_X_filter,action,test_size = 0.2,random_state=random_state)

				batch_size = 512
				train_X_total = np.reshape(train_X_total, (-1, 28,28,1))#size))
				test_X_total = np.reshape(test_X_total, (-1, 28,28,1))#size))
				train_dataset = tf.data.Dataset.from_tensor_slices((train_X_total, train_y_total))
				train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

				val_dataset = tf.data.Dataset.from_tensor_slices((test_X_total, test_y_total))
				val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)

				loss_train, loss_test, acc_train, acc_test = train(train_dataset,val_dataset)#,model)#$#keras_models[m])

				loss_train_list = []
				for p in range(len(loss_train)):
					loss_train_list.append(loss_train[p].numpy())

				loss_test_list = []
				for p in range(len(loss_test)):
					loss_test_list.append(loss_test[p].numpy())

				acc_train_list = []
				for p in range(len(acc_train)):
					acc_train_list.append(acc_train[p].numpy())

				acc_test_list = []
				for p in range(len(acc_test)):
					acc_test_list.append(acc_test[p].numpy())

				#history_total.append(history)
				# with open('loss_train_' + model_name + '_' + cost_name + '_' + further_name + '_' + save_name + '_' + str(coeff) + '_' + str(i+1) + '_' + str(random_state) + '_' + str(seed_value) + '_' + str(training) + '.txt', 'w') as f:
				# 	f.write(json.dumps(str(loss_train_list)))

				# with open('loss_test_' + model_name + '_' + cost_name + '_' + further_name + '_' + save_name + '_' + str(coeff) + '_' + str(i+1) + '_' + str(random_state) + '_' + str(seed_value) + '_' + str(training) + '.txt', 'w') as f:
				# 	f.write(json.dumps(str(loss_test_list)))

				# with open('acc_train_' + model_name + '_' + cost_name + '_' + further_name + '_' + save_name + '_' + str(coeff) + '_' + str(i+1) + '_' + str(random_state) + '_' + str(seed_value) + '_' + str(training) + '.txt', 'w') as f:
				# 	f.write(json.dumps(str(acc_train_list)))

				# with open('acc_test_' + model_name + '_' + cost_name + '_' + further_name + '_' + save_name + '_' + str(coeff) + '_' + str(i+1) + '_' + str(random_state) + '_' + str(seed_value) + '_' + str(training) + '.txt', 'w') as f:
				# 	f.write(json.dumps(str(acc_test_list)))
				
				weight_name = 'mnist_lr_' + model_name + '_' + cost_name + '_' + further_name + '_' + save_name + '_' + str(coeff) + '_' + str(i+1) + '_' + str(random_state) + '_' + str(seed_value) + '_' + str(training) +  '.h5'
				model.save_weights(weight_name)

				tree_q = []
				tree_N = []

		if retrain_sk:

			if (i+1) % retrain_step == 0:

				print('Retraining logistic regression')

				# node_total_retrain_transform = []
				# for t in range(len(node_total_retrain)):
				# 	state_retrain = np.array(node_total_retrain[t])
				# 	if cost_name == 'quadratic':
				# 		if further_name == 'end':
				# 			state_retrain = np.array([quad_end(cost(state_retrain)) if state_retrain[i] == -1.0 else state_retrain[i] for i in range(len(state_retrain))])
				# 		else:
				# 			state_retrain = np.array([quad_beg(cost(state_retrain)) if state_retrain[i] == -1.0 else state_retrain[i] for i in range(len(state_retrain))])
				# 	elif cost_name == 'linear':
				# 		state_retrain = np.array([linear_function(cost(state_retrain)) if state_retrain[i] == -1.0 else state_retrain[i] for i in range(len(state_retrain))])
				# 	else:
				# 		state_retrain = np.array([constant() if state_retrain[i] == -1.0 else state_retrain[i] for i in range(len(state_retrain))])
				# 	node_total_retrain_transform.append(state_retrain)

				#node_total_retrain_total = pd.DataFrame(node_total_retrain)

				#X_train_z_copy = X_train_z_copy.reshape((-1,784))

				#X_train_z_copy = pd.DataFrame(X_train_z_copy)

				#X_train_z_new = pd.concat([pd.DataFrame(X_train_z_copy),node_total_retrain_total],ignore_index = True)

				if purge_time > 1:
					y_train_new_z = []
					for k in range(purge_step*(purge_time-1),i+1):
						y_train_new_z.append(y_train_copy[k])
				else:
					y_train_new_z = []
					for k in range(i+1):
						y_train_new_z.append(y_train_copy[k])

				#y_train_new_z = pd.DataFrame(y_train_new_z)

				#y_train_new_z = pd.concat([pd.DataFrame(y_train_copy),y_train_new_z],ignore_index=True)
				
				#model_sk.fit(X_train_z_new, y_train_new_z)

				lst_to_be_added = list(set([i for i in range(10)]) - set(y_train_new_z))
				indx_to_be_added = []
				if lst_to_be_added:
					for w in range(len(lst_to_be_added)):
						l = np.where(y_train_new == lst_to_be_added[w])
						indx_to_be_added.append(l[0][0])
						y_train_new_z.append(lst_to_be_added[w])

				print(indx_to_be_added)

				new_nodes = []

				for b in range(len(indx_to_be_added)):
				    new_nodes.append(x_train_new[indx_to_be_added[b]].reshape((-1,784))[0])

				y_train_new_z_total = pd.DataFrame(y_train_new_z)

				print(y_train_new_z_total)

				y_train_new_z_total = pd.concat([pd.DataFrame(y_train_copy),y_train_new_z_total],ignore_index=True)

				retrain_node_transform_selected = pd.DataFrame(node_total_retrain)

				X_train_z_new = pd.concat([pd.DataFrame(X_train_z_copy.reshape((-1,784))),retrain_node_transform_selected,pd.DataFrame(new_nodes)],ignore_index = True)

				#print(time.time() - t1)

				#model_sk.fit(X_train_z_new, y_train_new_z_total)
				# model_zero.fit(node_retrain_total,y_train_new)

				batch_size = 512
				train_X_total = np.reshape(X_train_z_new, (-1, 28,28,1))
				test_X_total = np.reshape(x_test, (-1, 28,28,1))
				train_dataset = tf.data.Dataset.from_tensor_slices((train_X_total, y_train_new_z_total))
				train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

				val_dataset = tf.data.Dataset.from_tensor_slices((test_X_total, y_test))
				val_dataset = val_dataset.batch(batch_size)

				loss_train, loss_test, acc_train, acc_test = train_sk(train_dataset,val_dataset)
			# #history = model_sk.fit(X_train_transform, y_train_new, batch_size = 1, epochs = 301, validation_data = (X_test_total,y_test), verbose=1)

			# loss_train_list = []
			# for p in range(len(loss_train)):
			# 	loss_train_list.append(loss_train[p].numpy())

			# loss_test_list = []
			# for p in range(len(loss_test)):
			# 	loss_test_list.append(loss_test[p].numpy())

			# acc_train_list = []
			# for p in range(len(acc_train)):
			# 	acc_train_list.append(acc_train[p].numpy())

			# acc_test_list = []
			# for p in range(len(acc_test)):
			# 	acc_test_list.append(acc_test[p].numpy())

			# #history_total.append(history)
			# with open('loss_train_'+ str(i+1) + '_' + str(random_state) + '_1.txt', 'w') as f:
			# 	f.write(json.dumps(str(loss_train_list)))

			# with open('loss_test_'+ str(i+1) + '_' + str(random_state) + '_1.txt', 'w') as f:
			# 	f.write(json.dumps(str(loss_test_list)))

			# with open('acc_train_'+ str(i+1) + '_' + str(random_state) + '_1.txt', 'w') as f:
			# 	f.write(json.dumps(str(acc_train_list)))

			# with open('acc_test_'+ str(i+1) + '_' + str(random_state) + '_1.txt', 'w') as f:
			# 	f.write(json.dumps(str(acc_test_list)))

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

# def make(node,action,i,X_train): #action from 0 to 15

# 	lst = [[i] for i in range(0,784)]

# 	pairs = [[i,i] for i in range(0,784)]

# 	dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

# 	row_action = int(action/7.0) ##row action

# 	column_action = int(action - row_action*7.0) ##

# 	vec = np.array(node.tup)

# 	vec = vec.reshape((-1,28,28,1))

# 	vec[:,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)] = X_train[i,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1),:]

# 	acquired = []
# 	for k in range(49):
# 		row_action = int(k/7.0) ##row action
# 		column_action = int(k - row_action*7.0) ##
# 		if vec[:,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)].all() == X_train[i,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)].all():
# 			acquired.append(k) #already acquired

# 	is_terminal = len(acquired) == 49

# 	vec = vec.reshape((-1,784))

# 	return Tree(tuple(vec[0]), is_terminal) #new node

# def list_of_features(node,X_train):
    
# 	tup = np.array(node.tup)

# 	tup = tup.reshape((-1,28,28,1))

# 	lst = [[i] for i in range(0,784)]

# 	pairs = [[i,i] for i in range(0,784)]

# 	dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

# 	to_be_acquired = []

# 	for k in range(49):
# 		row_action = int(k/7.0) ##row action
# 		column_action = int(k - row_action*7.0) ##
# 		if tup[:,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)].all() == X_train[i,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)].all():
# 			to_be_acquired.append(k) #already acquired

# 	num_features = list(set(to_be_acquired))

# 	return num_features

# def action_probs(node,all_X,scores_total,i,X_train):

# 	def score(n):
# 		if scores_total.get(n) and all_X.get(n):
# 			return float(all_X.get(n) / scores_total.get(n))  # average reward
# 		else:
# 			return float("0")  # avoid unse

# 	acquired = list_of_features(node,X_train)
# 	total = [i for i in range(49)]

# 	scores = []
# 	for j in range(49):
# 		score_to_append = 0.0
# 		if total[j] in acquired:
# 			scores.append(0.0) #this hold all the time 
# 		else:
# 			score_to_append += score(node) #immediate reward (node)
# 			child = make(node,total[j],i,X_train)
# 			score_to_append += score(child) #first child
# 			scores.append(score_to_append)

# 	sum_scores = sum(scores)

# 	if sum_scores == 0.0:
# 		#scores_return = scores
# 		unacquired = list(set(total) - set(acquired))
# 		action = random.choice(unacquired)
# 	else:
# 		scores_return = [e/sum_scores for e in scores]
# 		common_indx = np.argwhere(scores_return==np.max(scores_return)).flatten().tolist()
# 		if len(common_indx) != 49:
# 			#action = np.argmax(scores_return)
# 		#else:
# 			action = random.choice(common_indx) #this can lead to same if only one

# 	return action

# retrain_step = 27

# def cost(state):

#     cost = 0

#     lst = [[i] for i in range(0,784)]

#     pairs = [[i,i] for i in range(0,784)]

#     dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

#     acquired = []
#     for k in range(len(lst)):
#         lst_to_be_checked = lst[k]
#         if state[lst_to_be_checked[0]] != 1.0:
#             acquired.append(lst_to_be_checked[0])

#     dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

#     num_features = list(set(dct_indx))

#     for i in range(len(num_features)):
#         cost += 1

#     return cost

# def slass(state,model):

#     state = np.asarray(state).flatten()

#     state = state.reshape((-1,784))##1,28,1))
#     classification = model.predict(state)
#     #classification = np.max(classification)
#     #classification = np.argmax(classification)
#     classification = classification.astype(float)
#     classification = classification[0]
#     classification = classification.flatten()
#     return classification

# import heapq

# import sklearn

# m,n,o,p = x_test.shape
# cost_total = []
# classification_total = []
# for i in range(m):

# 	print(i)

# 	tp = (1.0,)*784
# 	node = Tree(tup=tp, terminal=False) #the model is called multiple times

# 	act = []

# 	while True:

# 		cost_total.append(cost(node.tup))
# 		classification_total.append(slass(node.tup,model_sk))

# 		if node.terminal:
# 			break

# 		#node = tree.choose(node,i,X_train_z) #choose best score next state

# 		if integrated:

# 			tup_action = node.tup
# 			tup_action = np.array(tup_action).reshape(-1,size)

# 			logits = model(tup_action,training=False)
# 			logits = np.array(logits)
# 			logits = logits[0]

# 			action = heapq.nlargest(37, range(len(logits)), key=logits.__getitem__)
# 		#action = sorted(range(len(score_lst)), key=lambda x: score_lst[x])[::-1]
# 			st =  set(act)
# 			to_be = [ele for ele in action if ele not in st]
# 			action = to_be[0]

# 			act.append(action)

# 		#if interated:
# 		else:
# 			action = action_probs(node,result,result_N,i,x_test)

# 		node = make(node,action,i,x_test)

# 		#node_total.append(node.tup)

# name = 'node_train_' + cost_name + '_' + model_name + '_' + further_name + '_lr_physionet_mcts_smote_' + save_name + '_' + str(coeff) + '_' + str(random_state) + '_' + str(training) + '_1_' + str(retrain_step) + '_' + str(seed_value) + '_' + str(j) + '.txt'
# with open(name, 'w') as f:
# 	f.write(json.dumps(node_total))

# node_total = []

# if not integrated:

# 	# cost_name = 'quadratic'
# 	# model_name = 'retrain'
# 	# further_name = 'end'
# 	# save_name = 'standalone'
# 	# coeff = -70
# 	# random_state = 1
# 	# training = 100
# 	# retrain_step = 600
# 	# j = 2399
# 	# seed_value = 12321
    
# 	# import pickle
# 	# from multiprocessing import Pool

# 	# filename_Q = 'tree_Q_' + cost_name + '_' + model_name + '_' + further_name + '_lr_' + save_name + '_' + str(coeff) + '_'  + str(random_state) + '_' + str(training) + '_1_' + str(retrain_step) + '_' + str(seed_value) + '_' + str(j) + '.pickle'
# 	# with open(filename_Q, 'rb') as f:
# 	#     # Pickle the 'data' dictionary using the highest protocol available.
# 	# 	all_X = pickle.load(f)

# 	# filename_N = 'tree_N_' + cost_name + '_' + model_name + '_' + further_name + '_lr_' + save_name + '_' + str(coeff) + '_'  + str(random_state) + '_' + str(training) + '_1_' + str(retrain_step) + '_' + str(seed_value) + '_' + str(j) + '.pickle'
# 	# with open(filename_N, 'rb') as f:
# 	#     # Pickle the 'data' dictionary using the highest protocol available.
# 	# 	scores_total = pickle.load(f)

# 	def make_keys(key_iterable):

# 		keys_iterable, values = key_iterable

# 		global result#all_X
# 		global result_N#scores_total

# 		def make_child(vec,action,i): #make a single child

# 			lst = [[0,1,2,3,4,5,6,7,8,9], [10,11,12], [13,14,15], [16,17,18], [19,20,21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53]]

# 			dict_lst = {0: [0,1,2,3,4,5,6,7,8,9], 1: [10,11,12], 2: [13,14,15], 3: [16,17,18], 4: [19,20,21], 5: [22], 6: [23], 7: [24], 8: [25], 9: [26], 10: [27], 11: [28], 12: [29], 13: [30], 14: [31], 15: [32], 16: [33], 17: [34], 18: [35], 19: [36], 20: [37], 21: [38], 22: [39], 23: [40], 24: [41], 25: [42], 26: [43], 27: [44], 28: [45], 29: [46], 30: [47], 31: [48], 32: [49], 33: [50], 34: [51], 35: [52], 36: [53]}

# 		    #val = tuple([X_train_z.loc[i].iloc[dict_lst[action][j]] for j,value in enumerate(dict_lst[action])])

# 			val = tuple([X_train_z.iloc[i,dict_lst[action][j]] for j,value in enumerate(dict_lst[action])])

# 			tup = vec.tup
		    
# 			tup = tup[:dict_lst[action][0]] + val + tup[dict_lst[action][-1]+1:]

# 			acquired = []
# 			for k in range(len(lst)):
# 				lst_to_be_checked = lst[k]
# 				if lst_to_be_checked[0] <= 21 and tup[lst_to_be_checked[0]] != 1:
# 					acquired.append(lst_to_be_checked[0])
# 				elif lst_to_be_checked[0] > 21 and tup[lst_to_be_checked[0]] != -1:
# 					acquired.append(lst_to_be_checked[0])

# 			dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

# 			num_features = list(set(dct_indx))

# 			is_terminal = len(num_features) == 37

# 			return Tree(tup, is_terminal) #new node

# 		def list_of_features(node):
		    
# 			lst = [[0,1,2,3,4,5,6,7,8,9], [10,11,12], [13,14,15], [16,17,18], [19,20,21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53]]

# 			dict_lst = {0: [0,1,2,3,4,5,6,7,8,9], 1: [10,11,12], 2: [13,14,15], 3: [16,17,18], 4: [19,20,21], 5: [22], 6: [23], 7: [24], 8: [25], 9: [26], 10: [27], 11: [28], 12: [29], 13: [30], 14: [31], 15: [32], 16: [33], 17: [34], 18: [35], 19: [36], 20: [37], 21: [38], 22: [39], 23: [40], 24: [41], 25: [42], 26: [43], 27: [44], 28: [45], 29: [46], 30: [47], 31: [48], 32: [49], 33: [50], 34: [51], 35: [52], 36: [53]}

# 			acquired = []
# 			for k in range(len(lst)):
# 				lst_to_be_checked = lst[k]
# 				if lst_to_be_checked[0] <= 21 and node[lst_to_be_checked[0]] != 1:
# 					acquired.append(lst_to_be_checked[0])
# 				elif lst_to_be_checked[0] > 21 and node[lst_to_be_checked[0]] != -1:
# 					acquired.append(lst_to_be_checked[0])

# 			dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

# 			return list(set(dct_indx))

# 		def action_probs(node,all_X,scores_total,i):

# 			def score(n):
# 				if scores_total.get(n) and all_X.get(n):
# 					return float(all_X.get(n) / scores_total.get(n))  # average reward
# 				else:
# 					return float("0")  # avoid unse

# 			acquired = list_of_features(node.tup)
# 			total = [i for i in range(37)]

# 			scores = []
# 			for j in range(37):
# 				score_to_append = 0.0
# 				if total[j] in acquired:
# 					scores.append(0.0) #this hold all the time 
# 				else:
# 					score_to_append += score(node) #immediate reward (node)
# 					child = make_child(node,total[j],i)
# 					score_to_append += score(child) #first child
# 					scores.append(score_to_append)

# 			sum_scores = sum(scores)

# 			if sum_scores == 0.0:
# 				scores_return = scores
# 			else:
# 				scores_return = [e/sum_scores for e in scores]

# 			return scores_return

# 		keys = []
# 		scores = []
# 		indx_done = []
# 		#i = 0
# 		#j = 0

# 		key = keys_iterable.tup

# 		print(key)

# 		value_iterable = values#[i]

# 		#feature_values = [value for i, value in enumerate(key) if value != -1] #based on real-valued features?
# 		#if feature_values:
# 		#need to keep the position
# 		sum_lst = []
# 		for z in range(len(X_train_z)):
# 			sum_lst.append(sum([x == y for (x,y) in zip(list(X_train_z.iloc[z,:]),key)]))#feature_values)]))

# 		#print(sum_lst)
# 		if np.mean(sum_lst) > 10:
# 			indx = np.argwhere(sum_lst == np.amax(sum_lst)).flatten().tolist()
# 			indx_copy = indx.copy()
# 			indx_done_copy = indx_done.copy()
# 			while indx:
# 				min_indx = np.min(indx)
# 				if min_indx not in indx_done:
# 					indx_done.append(min_indx)
# 					break
# 				else:
# 					indx_to_be_appended = np.argwhere(indx==min_indx).flatten().tolist() #can be multiple
# 					indx.pop(indx_to_be_appended[0])
# 			if not indx and len(indx_done) == len(indx_done_copy):
# 				while indx_copy:
# 					min_indx_copy = np.min(indx_copy)
# 					count = indx_done.count(min_indx_copy)
# 					if count <= 1000:
# 						indx_done.append(min_indx_copy)
# 						break
# 					else:
# 						indx_copy_to_be_appended = np.argwhere(indx_copy == min_indx_copy).flatten().tolist()
# 						indx_copy.pop(indx_copy_to_be_appended[0])
# 		else:
# 			indx = []

# 		if indx_done:
# 			keys.append(key)
# 			scores.append(action_probs(keys_iterable,result,result_N,indx_done[-1]))

# 		print(scores)
# 		return keys,scores

# 	p = Pool(8)

# 	from itertools import islice

# 	def take(n, iterable):
# 		"Return first n items of the iterable as a list"
# 		return list(islice(iterable, n))

# 	n = 4000

# 	n_items = take(n, result.items())
# 	n_items = dict(n_items)
	
# 	finalValue = list(p.map(make_keys, n_items.items()))

# 	keys_train = []
# 	scores_train = []
# 	for k in range(len(finalValue)):
# 		if finalValue[k][0]:
# 			keys_train.append(list(finalValue[k][0][0]))
# 			scores_train.append(finalValue[k][1][0])

# 	keys_train = np.array(keys_train)
# 	scores_train = np.array(scores_train)

# 	all_X = [] 
# 	all_Y = []
# 	#import random
# 	action = []
# 	for i in range(len(scores_train)):
# 		criterion = list(scores_train[i])
# 		common_indx = np.argwhere(criterion==np.max(criterion)).flatten().tolist()
# 		if len(common_indx) != 37:
# 			action.append(random.choice(common_indx)) #this can lead to same if only one
# 			all_X.append(keys_train[i])
# 	    #print(i)

# 	all_X = np.array(all_X)
# 	action = np.array(action)

# 	import tensorflow as tf
# 	from tensorflow import keras
# 	import numpy as np
# 	import pickle
# 	import pandas as pd 
# 	from keras.regularizers import l2

# 	size_dim = 54

# 	initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal')
# 	inputs = keras.Input(shape=(size_dim,), name="digits")

# 	x1 = keras.layers.Dense(1024, activation="relu",kernel_initializer=initializer,kernel_regularizer = l2(0.01))(inputs)
# 	x2 = keras.layers.Dense(512, activation="relu",kernel_initializer=initializer,kernel_regularizer = l2(0.01))(x1)
# 	x3 = keras.layers.Dense(256, activation="relu",kernel_initializer=initializer,kernel_regularizer = l2(0.01))(x2)
# 	x4 = keras.layers.Dense(128, activation="relu",kernel_initializer=initializer, kernel_regularizer = l2(0.01))(x3)
# 	x5 = keras.layers.Dense(64, activation="relu",kernel_initializer=initializer, kernel_regularizer = l2(0.01))(x4)
# 	#x6 = keras.layers.Dense(32, activation="relu",kernel_initializer=initializer, kernel_regularizer = l2(0.01))(x5)
# 	outputs = keras.layers.Dense(37,activation ='softmax',name="predictions",dtype='float64')(x5)
# 	model = keras.Model(inputs=inputs, outputs=outputs)

# 	from sklearn.model_selection import train_test_split
# 	X_train, X_test, Y_train, Y_test = train_test_split(all_X, action, test_size=0.2, random_state=1)
# 	X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1)

# 	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])#, tf.keras.metrics.FalseNegatives()])
# 	#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10)
# 	history = model.fit(x=X_train, y=Y_train, validation_data = (X_val,Y_val), batch_size=512, epochs=3000, verbose=1)#, callbacks=[es])

# 	print("Evaluate on test data")
# 	results = model.evaluate(X_test, Y_test, batch_size=512)
# 	print("test loss, test acc:", results)

# 	def num_features(tp):

# 		lst = [[0,1,2,3,4,5,6,7,8,9], [10,11,12], [13,14,15], [16,17,18], [19,20,21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53]]

# 		dict_lst = {0: [0,1,2,3,4,5,6,7,8,9], 1: [10,11,12], 2: [13,14,15], 3: [16,17,18], 4: [19,20,21], 5: [22], 6: [23], 7: [24], 8: [25], 9: [26], 10: [27], 11: [28], 12: [29], 13: [30], 14: [31], 15: [32], 16: [33], 17: [34], 18: [35], 19: [36], 20: [37], 21: [38], 22: [39], 23: [40], 24: [41], 25: [42], 26: [43], 27: [44], 28: [45], 29: [46], 30: [47], 31: [48], 32: [49], 33: [50], 34: [51], 35: [52], 36: [53]}

# 		acquired = []
# 		for j in range(len(lst)):
# 			lst_to_be_checked = lst[j]
# 			if lst_to_be_checked[0] <= 21 and tp[lst_to_be_checked[0]] != 1:
# 				acquired.append(lst_to_be_checked[0])
# 			elif lst_to_be_checked[0] > 21 and tp[lst_to_be_checked[0]] != -1:
# 				acquired.append(lst_to_be_checked[0])

# 		dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

# 		num_features = list(set(dct_indx))

# 		return len(acquired)

# 	def make(vec,action,i):#,all_X_total): #action from 0 to 15
	        
# 		lst = [[0,1,2,3,4,5,6,7,8,9], [10,11,12], [13,14,15], [16,17,18], [19,20,21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53]]

# 		dict_lst = {0: [0,1,2,3,4,5,6,7,8,9], 1: [10,11,12], 2: [13,14,15], 3: [16,17,18], 4: [19,20,21], 5: [22], 6: [23], 7: [24], 8: [25], 9: [26], 10: [27], 11: [28], 12: [29], 13: [30], 14: [31], 15: [32], 16: [33], 17: [34], 18: [35], 19: [36], 20: [37], 21: [38], 22: [39], 23: [40], 24: [41], 25: [42], 26: [43], 27: [44], 28: [45], 29: [46], 30: [47], 31: [48], 32: [49], 33: [50], 34: [51], 35: [52], 36: [53]}

# 		val = tuple([X_train_z.iloc[i,dict_lst[action][j]] for j,value in enumerate(dict_lst[action])])

# 		tup = tuple(vec[0][:dict_lst[action][0]]) + val + tuple(vec[0][dict_lst[action][-1]+1:])

# 		tup = np.array(tup)

# 		num = num_features(tup)

# 		done = num == 37

# 		return tup, done

# 	import heapq
# 	total = []
# 	m,n = X_test_z.shape
# 	for i in range(m):
# 		tp = (1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1)
# 		tp = np.array(tp).reshape(-1,size_dim)
# 		act = []
# 		print(i)
# 		done = False
# 		while True:

# 			total.append(tp)

# 			if done:
# 				break

# 			logits = model(tp,training = False)
# 			logits = np.array(logits)
# 			logits = logits[0]
# 			action = heapq.nlargest(37, range(len(logits)), key=logits.__getitem__)

# 			#action = sorted(range(len(score_lst)), key=lambda x: score_lst[x])[::-1]

# 			st =  set(act)

# 			to_be = [ele for ele in action if ele not in st]

# 			action = to_be[0]

# 			act.append(action)

# 			tp, done = make(tp,action,i)

# 			tp = tp.reshape(-1,size_dim)

# 		print(act)

# 	total_list = []
# 	for i in range(len(total)):
# 		node = tuple(np.array(total[i][0],dtype=np.float64))
# 		total_list.append(node)

# 	name = 'node_test_' + model_name + '_' + cost_name + '_' + further_name + '_lr_mcts_physionet_smote_' + save_name + '_' + str(coeff) + '_' + str(random_state) + '_' + str(seed_value) + '_' + str(training) + '.txt'
# 	with open(name, 'w') as f:
# 		f.write(json.dumps(total_list))

# if integrated:

# 	def num_features(tp):

# 		lst = [[0,1,2,3,4,5,6,7,8,9], [10,11,12], [13,14,15], [16,17,18], [19,20,21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53]]

# 		dict_lst = {0: [0,1,2,3,4,5,6,7,8,9], 1: [10,11,12], 2: [13,14,15], 3: [16,17,18], 4: [19,20,21], 5: [22], 6: [23], 7: [24], 8: [25], 9: [26], 10: [27], 11: [28], 12: [29], 13: [30], 14: [31], 15: [32], 16: [33], 17: [34], 18: [35], 19: [36], 20: [37], 21: [38], 22: [39], 23: [40], 24: [41], 25: [42], 26: [43], 27: [44], 28: [45], 29: [46], 30: [47], 31: [48], 32: [49], 33: [50], 34: [51], 35: [52], 36: [53]}

# 		acquired = []
# 		for j in range(len(lst)):
# 			lst_to_be_checked = lst[j]
# 			if lst_to_be_checked[0] <= 21 and tp[lst_to_be_checked[0]] != 1:
# 				acquired.append(lst_to_be_checked[0])
# 			elif lst_to_be_checked[0] > 21 and tp[lst_to_be_checked[0]] != -1:
# 				acquired.append(lst_to_be_checked[0])

# 		dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

# 		num_features = list(set(dct_indx))

# 		return len(acquired)

# 	def make(vec,action,i):#,all_X_total): #action from 0 to 15
	        
# 		lst = [[0,1,2,3,4,5,6,7,8,9], [10,11,12], [13,14,15], [16,17,18], [19,20,21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53]]

# 		dict_lst = {0: [0,1,2,3,4,5,6,7,8,9], 1: [10,11,12], 2: [13,14,15], 3: [16,17,18], 4: [19,20,21], 5: [22], 6: [23], 7: [24], 8: [25], 9: [26], 10: [27], 11: [28], 12: [29], 13: [30], 14: [31], 15: [32], 16: [33], 17: [34], 18: [35], 19: [36], 20: [37], 21: [38], 22: [39], 23: [40], 24: [41], 25: [42], 26: [43], 27: [44], 28: [45], 29: [46], 30: [47], 31: [48], 32: [49], 33: [50], 34: [51], 35: [52], 36: [53]}

# 		val = tuple([X_test_z.iloc[i,dict_lst[action][j]] for j,value in enumerate(dict_lst[action])])

# 		tup = tuple(vec[0][:dict_lst[action][0]]) + val + tuple(vec[0][dict_lst[action][-1]+1:])

# 		tup = np.array(tup)

# 		num = num_features(tup)

# 		done = num == 37

# 		return tup, done

# 	import heapq
# 	total = []
# 	m,n = X_test_z.shape
# 	for i in range(m):
# 		tp = (1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1)
# 		tp = np.array(tp).reshape(-1,size)
# 		act = []
# 		print(i)
# 		done = False
# 		while True:

# 			total.append(tp)

# 			if done:
# 				break

# 			logits = model(tp,training = False)
# 			logits = np.array(logits)
# 			logits = logits[0]
# 			action = heapq.nlargest(37, range(len(logits)), key=logits.__getitem__)

# 			st =  set(act)

# 			to_be = [ele for ele in action if ele not in st]

# 			action = to_be[0]

# 			act.append(action)

# 			tp, done = make(tp,action,i)
			
# 			tp = tp.reshape(-1,size)

# 		print(act)

# 	total_list = []
# 	for i in range(len(total)):
# 		node = tuple(np.array(total[i][0],dtype=np.float64))
# 		total_list.append(node)

# 	name = 'node_test_' + model_name + '_' + cost_name + '_' + further_name + '_lr_mcts_physionet_smote_' + save_name + '_' + str(coeff) + '_' + str(random_state) + '_' + str(seed_value) + '_' + str(training) + '_' + str(retrain_step) + '.txt'
# 	with open(name, 'w') as f:
# 		f.write(json.dumps(total_list))

# tp = (0.0,)*784
# node = Tree(tp,terminal=False)
# i = 0
# act = []
# while True:
# 	i += 1
# 	print(i)
# 	if len(act) == 49: #node.terminal:
# 		break
# 	node, action = node.find_random_child(0,x_train_new,act)
# 	act.append(action)
# 	print(node.tup)