from multiprocessing import Pool
from monte_carlo_tree_search_mo_mnist import MCTS, Node
from tree_mo_mnist import Tree
from sklearn.linear_model import LogisticRegression as sk
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import time
import collections, functools, operator 
import pickle
import json

from parameters_MCTS_MO import*

from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import tensorflow as tf

import random

#pd.options.mode.chained_assignment = None  # default='warn'

import json
import matplotlib.pyplot as plt

import heapq

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

filename = 'finalized_model_mnist_' + str(random_state) + '_' + str(seed_value) + '.sav'
model_zero = pickle.load(open(filename, 'rb'))

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.regularizers import l1_l2
# import pickle 
# tf.random.set_seed(seed_value)

# initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal')
# inputs = keras.Input(shape=(27,), name="digits")
# x1 = layers.Dense(64,  kernel_regularizer=l1_l2(l1=1e-1, l2=1e-1), activation="relu",kernel_initializer=initializer)(inputs)
# x1_outputs = layers.Dropout(0.7)(x1, training=True)
# x2 = layers.Dense(32,  kernel_regularizer=l1_l2(l1=1e-1, l2=1e-1), activation="relu",kernel_initializer=initializer)(x1_outputs)
# x2_outputs = keras.layers.Dropout(0.7)(x1, training=True)
# x3 = layers.Dense(16,  kernel_regularizer=l1_l2(l1=1e-1, l2=1e-1), activation="relu",kernel_initializer=initializer)(x2_outputs)
# outputs = layers.Dense(2,activation ='softmax',name="predictions",dtype='float64')(x3)
# model_zero = keras.Model(inputs=inputs, outputs=outputs)

# if retrain_sk:
# 	# # # Instantiate an optimizer.
# 	lr = 1e-5
# 	epochs = 101

# 	# Instantiate a loss function.
# 	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# 	# Prepare the metrics.
# 	train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
# 	val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# 	@tf.function
# 	def train_sk_step(x, y):
# 		with tf.GradientTape() as tape:
# 		    logits = model_zero(x, training=True)
# 		    loss_value = loss_fn(y, logits)
# 		grads = tape.gradient(loss_value, model_zero.trainable_weights)
# 		optimizer.apply_gradients(zip(grads, model_zero.trainable_weights))
# 		train_acc_metric.update_state(y, logits)
# 		return loss_value

# 	@tf.function
# 	def test_sk_step(x, y):
# 		val_logits = model_zero(x, training=False)
# 		loss_value = loss_fn(y,val_logits)
# 		val_acc_metric.update_state(y, val_logits)
# 		return loss_value

# 	def train_sk(train_dataset,val_dataset):
# 		loss_train = []
# 		acc_train = []
# 		loss_test = []
# 		acc_test = []
# 		for epoch in range(epochs):
# 		    print("\nStart of epoch %d" % (epoch,))
# 		    start_time = time.time()
# 		    loss_train_epoch = 0.0
# 		    # Iterate over the batches of the dataset.
# 		    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
# 		        loss_value = train_sk_step(x_batch_train, y_batch_train)
# 		        loss_train_epoch += loss_value
# 		    loss_train_epoch /= (step+1)
# 		    loss_train.append(loss_train_epoch)

# 		    # Display metrics at the end of each epoch.
# 		    train_acc = train_acc_metric.result()
# 		    acc_train.append(train_acc)

# 		    print("Training acc over epoch: %.4f" % (float(train_acc),))
# 		    print("Training loss over epoch: %.4f" % (float(loss_train_epoch),))

# 		    # Reset training metrics at the end of each epoch
# 		    train_acc_metric.reset_states()

# 		    loss_test_epoch = 0.0
# 		    # Run a validation loop at the end of each epoch.
# 		    for step_val, (x_batch_val, y_batch_val) in enumerate(val_dataset):
# 		        loss_value_test = test_sk_step(x_batch_val, y_batch_val)
# 		        loss_test_epoch += loss_value_test
# 		    loss_test_epoch /= (step_val+1)
# 		    loss_test.append(loss_test_epoch)

# 		    val_acc = val_acc_metric.result()

# 		    acc_test.append(val_acc)

# 		    print("Validation acc: %.4f" % (float(val_acc),))
# 		    print("Validation loss: %.4f" % (float(loss_test_epoch),))

# 		    val_acc_metric.reset_states()

# 		    print("Time taken: %.2fs" % (time.time() - start_time))

# 		return loss_train, loss_test, acc_train, acc_test

# import pickle

# if pretrain_model or fit_model:
# 	#filename = 'finalized_model_hf_smote' + str(random_state) + '_' + str(seed_value) + '.sav'
# 	filename = 'hf' + '_' + str(random_state) + '_' + str(seed_value)# + '_quadratic_end'
# 	weight_name = filename + '.h5'

# 	# # # NOW set the trainable weights of the model
# 	model_zero.load_weights(weight_name)

# elif retrain_model:
# 	filename = 'hf' + '_' + str(random_state) + '_' + str(seed_value)# + '_quadratic_end'
	
# 	weight_name = filename + '.h5'

# 	opt_name = filename + '_optimizer.pkl'

# 	with open(opt_name, 'rb') as f:
# 		weight_values = pickle.load(f)

# 	grad_vars = model_zero.trainable_weights

# 	optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

# 	zero_grads = [tf.zeros_like(w) for w in grad_vars]

# 	# Apply gradients which don't do nothing with Adam
# 	optimizer.apply_gradients(zip(zero_grads, grad_vars))

# 	# Set the weights of the optimizer
# 	optimizer.set_weights(weight_values)

# elif random_model:
# 	if beg or end:
# 		filename = 'hf_random' + '_' + str(coeff) + '_' + str(random_state) + '_' + str(seed_value) + '_' + cost_name + '_' + further_name# + '.h5'#quadratic_end'
# 		weight_name = filename + '.h5'
# 		model_zero.load_weights(weight_name)

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

    # model = sk(random_state = seed_value,max_iter=10000)

    # test_tup = np.array((1.0e-6,)*784)#.reshape(1,-1)
    # test_tup = np.repeat(test_tup,49)#pd.DataFrame(y_test[:2])
    # test_tup = test_tup.reshape(-1,784)

    # test_logits = np.array([i for i in range(49)])

    # model.fit(test_tup,test_logits)

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
                    #hv_vol = HyperVolume(referencePoint)
                    score = hypervolume_lst(P)#/self.N[n]

                    return score#hypervolume(self.local_P[n])/self.N[n] #self.Q[n] / self.N[n]  # average reward
                else:
                    return float("0")  # avoid unseen

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

indx = []
for j in range(10):
    l = np.where(y_train == j)
    for j in range(2):#len(l[0])):
        indx.append(l[0][j])
    #l = l[0][0]
    #indx.append(l)

x_train_new = []
for j in range(len(indx)):
    x = x_train[indx[j],:]
    x_train_new.append(np.array(x))

x_train_new = np.array(x_train_new)
y_train_new = np.array(2*[i for i in range(10)])

x_train_new_linear = x_train_new.reshape((-1,784))
x_train_new_linear = pd.DataFrame(x_train_new_linear)

y_train_new = np.sort(y_train_new)

m,n,o,p = x_train_new.shape

node_total_total = []
tree_q_total = []
tree_N_total = []
purge_step = int(m/2.0)
save_step = purge_step
size = 784

act_total = []
t0 = time.time()
for v in range(num_epochs):
	node_total = []
	node_total_retrain = []
	hv_total = []
	hv_hv_total = []
	P_total = []
	tree_q = []
	tree_N = []
	X_train_z_copy = x_train_new.copy()
	y_train_copy = y_train_new.copy()
	retrain_time = 1
	purge_time = 1
	#print(j)
	for i in range(10):#m):
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

			#print(hv_node)

			nodes.append(node.tup)

			#num_node += 1

			#hv_node = [i/tree.N[node] for i in tree.local_P[node]]
			# if tree.local_P[node][1] < 1.0:
			# 	hv_node_save = [i for i in tree.local_P[node]]
			# else:
			# 	hv_node_save = [i/tree.N[node] for i in tree.local_P[node]]

			if node.terminal:
				if isinstance(tree.local_P[node][0],float):
					#if tree.local_P[node][1] < 1.0:
					#	hv_node_save = [i for i in tree.local_P[node]]
					#else:
					hv_node_save = [i / (tree.N[node]+1) for i in tree.local_P[node]]
				else:
					hv_node_save_lst = []
					for j in range(len(tree.local_P[node])):
					#	if tree.local_P[node][j][1] < 1.0:
					#		hv_node_save_lst.append([i for i in tree.local_P[node][j]])
					#	else:
						hv_node_save_lst.append([i / (tree.N[node]+1) for i in tree.local_P[node][j]])
					hv_node_save = flatten(hv_node_save_lst)
					row = int(len(hv_node_save)/2.0)
					col = 2
					hv_node_save = [hv_node_save[col*i : col*(i+1)] for i in range(row)]

				#if isinstance(hv_node_save[0],float):
				hv_hv.append(hypervolume(hv_node))
				hv.append(hv_node)
				#else:
					#print('yest')
				#	hv_hv.append(score(node))#hypervolume_lst(hv_node_save))
				#	hv.append(hv_node_save)
				break

			for z in range(training):
				print(z) 
				tree.train(node,i,model_zero,x_train_new,y_train_new)

			#node, score_node = tree.choose(node,i,X_train_z) #choose best score next state

			if integrated:

				tup_action = node.tup
				tup_action = np.array(tup_action).reshape((-1,28,28,1))#size)
				#logits = model.predict_proba(tup_action)
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
				node, score_node = tree.choose(node,i,x_train_new) #choose best score next state

			if isinstance(tree.local_P[node][0],float):
				#if tree.local_P[node][1] < 1.0:
			#		hv_node_save = [i for i in tree.local_P[node]]
		#		else:
				hv_node_save = [i / (tree.N[node]+1) for i in tree.local_P[node]]
			else:
				hv_node_save_lst = []
				for j in range(len(tree.local_P[node])):
				#	if tree.local_P[node][j][1] < 1.0:
				#		hv_node_save_lst.append([i for i in tree.local_P[node][j]])
				#	else:
					hv_node_save_lst.append([i / (tree.N[node]+1) for i in tree.local_P[node][j]])
				hv_node_save = flatten(hv_node_save_lst)
				row = int(len(hv_node_save)/2.0)
				col = 2
				hv_node_save = [hv_node_save[col*i : col*(i+1)] for i in range(row)]

			#print(tree.local_P[node])
			#print(hv_node_save)
			#hv_node = [-node.cost(),node.f1(model_zero)]
			#hv_node = np.mean(hv_node,axis=0)
			#hv_node = list(hv_node)
			#hv.append(hv_node)
			#hyperVolume = HyperVolume(referencePoint)
			#hv_hv.append(hyperVolume.compute([hv_node]))
			dict_node = {}
			dict_node[node] = hv_node_save
			tree_q.append(dict_node)

			hv_hv.append(hypervolume(hv_node))
			hv.append(hv_node)

			print(hv)

			#if isinstance(hv_node_save[0],float):
			#	hv_hv.append(hypervolume(hv_node_save))
			#	hv.append(hv_node_save)
			#else:
				#print('yest')
			#	hv_hv.append(hypervolume_lst(hv_node_save))
			#	hv.append(hv_node_save)
				#hv_node_save = np.mean(hv_node_save,axis=0)
				#hv_node_Save = list(hv_node_save)

			#print(node.terminal)

			#print(node)
			#print(num)
			#num_node += 1
			P = [x for x in tree.global_P if x]
			P_node = {}
			P_node[node] = P
			tree_N.append(P_node)
			print(act)
		#act_total.append(act)
		hv_total.append(hv)
		hv_hv_total.append(hv_hv)
		#plt.plot(hv_hv_total[i])
		#plt.show()
		node_total.append(nodes)
		P = [x for x in tree.global_P if x]
		#P_node = {}
		#P_node[node] = P
		#tree_N.append(P_node)
		if isinstance(P[0],float):
			P_total.append(P)
		else:
			P = [list(x) for x in set(tuple(x) for x in P)]
			P_total.append(P)

		#print(P)
		#for i in range(len(P)):
		#	P[i][0] /= training
		#	P[i][1] /= training
		#for k in range(len(P)):
		#	P_total.append(P[k])
		#print(P[0])
		#P[0][0] /= training
		#P[0][1] /= training
		#P_total.append(P[0])
		#print(P_total)

		#print(hypervolume(P))
		#tree_q.append(dict_node)
		#tree_N.append(tree.N)

		if retrain_model:

			tp = (1.0e-6,)*784
			node_retrain = Tree(tup=tp, terminal=False) #the model is called multiple times
			num = np.random.randint(1,49)
			num_node = 1
			tree_retrain = MCTS() #for every new sample

			while num_node<num:
			# 	#based on the given node, find the MCTS next state
				if node_retrain.terminal:
					break

				for k in range(50): 
					tree_retrain.train(node_retrain,i,model_zero,x_train_new,y_train_new)

				node_retrain, score_retrain = tree_retrain.choose(node_retrain,i,x_train_new) #choose best score next state

				num_node += 1
			
			node_total_retrain.append(node_retrain.tup)

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

				# model.fit(train_X_total.reshape((-1,784)),train_y_total)

				# predictions = model.predict(train_X_total.reshape((-1,784)))
				# print(metrics.f1_score(train_y_total,predictions,average='micro'))

				# predictions = model.predict(test_X_total.reshape((-1,784)))
				# print(metrics.f1_score(test_y_total,predictions,average='micro'))

				weight_name = 'mnist_mo_lr_model_weights_' + model_name + '_' + save_name + '_' + further_name + '_' + str(random_state) + '_' + str(seed_value) + '.h5'
				model.save_weights(weight_name)
				#pickle.dump(model, open(filename, 'wb'))

				tree_q = []
				tree_N = []

		if retrain_sk:

			if (i+1) % retrain_step == 0:

				print('Retraining NN')

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

	if (v+1) % num_epochs == 0:

		print('Saving')

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

def cost(state,j):

	cost = 0
	total_cost = 784.0

	#tup = np.array(state.tup)

	state = state.reshape((-1,28,28,1))

	lst = [[i] for i in range(0,784)]

	pairs = [[i,i] for i in range(0,784)]

	dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

	to_be_acquired = []

	for z in range(49):
	   row_action = int(z/7.0) ##row action
	   column_action = int(z - row_action*7.0)
	   if np.allclose(state[:,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)],x_train_new[j,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)]):#,atol=1.0e-6):
	       to_be_acquired.append(z) #already acquired

	num_features = list(set(to_be_acquired))

	for i in range(len(num_features)):
	   cost += 16


	return cost

class_prob = []
costs_nodes = []
for j in range(len(node_total)):
	class_prob_ind = []
	costs_nodes_ind = []
	for k in range(len(node_total[j])):
		indx = int(j)
		print(indx)
		node_check = np.array(node_total[j][k])
		#node_check = [quad_end()]
    #classification = model_zero.predict(node_check.reshape(1,-1))
		costs_nodes.append(-cost(node_check,indx)/784.0)
		#costs_nodes_ind.append(-cost(node_check)/41.0)
		#for z in range(len(node_check)):
		#	if node_check[z] == -1.0:
		#		node_check[z] = 0.0
		prob = model_zero.predict_proba(node_check.reshape(1,-1)).flatten() # this should be quad_end
		classification = prob[y_train_new[indx]]
    #classification = prob[int(classification[0])]
		#if cost(node_check) != 0.0:
		#class_prob_ind.append(classification)
		class_prob.append(classification)
	#class_prob.append(class_prob_ind)
	#costs_nodes.append(costs_nodes_ind)

import collections

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

P_total = flatten(P_total)
row = int(len(P_total)/2.0)
col = 2
P_total = [P_total[col*i : col*(i+1)] for i in range(row)]

hv_total = flatten(hv_total)
row = int(len(hv_total)/2.0)
col = 2
hv_total = [hv_total[col*i : col*(i+1)] for i in range(row)]

# paretoPoints_total = []
# for j in range(len(P_total)):
# 	new_list = P_total[j].copy()
# 	paretoPoints, dominatedPoints = simple_cull(new_list,dominates)
# 	paretoPoints = list(paretoPoints)
# 	for i in range(len(paretoPoints)):
# 		paretoPoints[i] = list(paretoPoints[i])
# 	paretoPoints.sort()
# 	paretoPoints_total.append(paretoPoints)

# paretoPoints_sim_total = []
# for j in range(len(costs_nodes)):
# 	new_lst = [list(x) for x in zip(costs_nodes[j],class_prob[j])]
# 	paretoPoints_node, dominatedPoints_node = simple_cull(new_lst,dominates)
# 	paretoPoints_node = list(paretoPoints_node)
# 	for i in range(len(paretoPoints_node)):
# 		paretoPoints_node[i] = list(paretoPoints_node[i])
# 	paretoPoints_node.sort()
# 	paretoPoints_sim_total.append(paretoPoints_node)

# pareto_x_total = []
# pareto_y_total = []
# for j in range(len(paretoPoints_total)):
# 	pareto_x = []
# 	pareto_y = []
# 	for i in range(len(paretoPoints_total[j])):
# 		pareto_x.append(paretoPoints_total[j][i][0])
# 		pareto_y.append(paretoPoints_total[j][i][1])
# 	pareto_x_total.append(pareto_x)
# 	pareto_y_total.append(pareto_y)

# pareto_x_sim_total = []
# pareto_y_sim_total = []
# for j in range(len(paretoPoints_sim_total)):
# 	pareto_x = []
# 	pareto_y = []
# 	for i in range(len(paretoPoints_sim_total[j])):
# 		pareto_x.append(paretoPoints_sim_total[j][i][0])
# 		pareto_y.append(paretoPoints_sim_total[j][i][1])
# 	pareto_x_sim_total.append(pareto_x)
# 	pareto_y_sim_total.append(pareto_y)

P_total_copy = P_total.copy()

paretoPoints, dominatedPoints = simple_cull(P_total,dominates)
paretoPoints = list(paretoPoints)

for i in range(len(paretoPoints)):
	paretoPoints[i] = list(paretoPoints[i])

paretoPoints.sort()

pareto_x = []
pareto_y = []
for k in range(len(paretoPoints)):
	pareto_x.append(paretoPoints[k][0])
	pareto_y.append(paretoPoints[k][1])

# paretoPoints_sim, dominatedPoints_sim = simple_cull(hv_total,dominates)
# paretoPoints_sim = list(paretoPoints_sim)

# for i in range(len(paretoPoints_sim)):
# 	paretoPoints_sim[i] = list(paretoPoints_sim[i])

# paretoPoints_sim.sort()

# pareto_x_sim = []
# pareto_y_sim = []
# for k in range(len(paretoPoints_sim)):
# 	pareto_x_sim.append(paretoPoints_sim[k][0])
# 	pareto_y_sim.append(paretoPoints_sim[k][1])

new_lst = [list(x) for x in zip(costs_nodes, class_prob)]

paretoPoints_node, dominatedPoints_node = simple_cull(new_lst,dominates)
paretoPoints_node = list(paretoPoints_node)

for i in range(len(paretoPoints_node)):
	paretoPoints_node[i] = list(paretoPoints_node[i])

paretoPoints_node.sort()

pareto_x_node = []
pareto_y_node = []
for k in range(len(paretoPoints_node)):
	pareto_x_node.append(paretoPoints_node[k][0])
	pareto_y_node.append(paretoPoints_node[k][1])

# cost_new = []
# for i in range(len(lst1)):
#     cost_node = lst1[i]/(-41.0)
#     cost_new.append(cost_node)

# new_lst_f1 =  [list(x) for x in zip(cost_new, score)]

# paretoPoints_f1, dominatedPoints_f1 = simple_cull(new_lst_f1,dominates)
# paretoPoints_f1 = list(paretoPoints_f1)

# for i in range(len(paretoPoints_f1)):
#     paretoPoints_f1[i] = list(paretoPoints_f1[i])

# paretoPoints_f1.sort()

# pareto_x_f1 = []
# pareto_y_f1 = []
# for k in range(len(paretoPoints_f1)):
#     pareto_x_f1.append(paretoPoints_f1[k][0])
#     pareto_y_f1.append(paretoPoints_f1[k][1])

#hv_total = flatten(hv_total)
#row = int(len(hv_total)/2.0)
#col = 2
#hv_total = [hv_total[col*i : col*(i+1)] for i in range(row)]

# pareto_x_sim = []
# pareto_y_sim = []
# for k in range(len(costs_nodes)):
# 	pareto_x_sim.append(costs_nodes[k])
# 	pareto_y_sim.append(class_prob[k])

# costs = list(set(pareto_x_sim))

# import random

# def find(lst, a):
#     result = []
#     for i, x in enumerate(lst):
#         if x == a:
#             result.append(i)
#     return result

# ave_prob = []
# ave_cost = []
# for i in range(len(costs)):
#     #if costs[i] != 0.0:
#     find_cost = find(pareto_x_sim,costs[i])
#     ave = []
#     print(len(find_cost))
#     if len(find_cost) > 10:
#         for j in range(len(find_cost)):
#             ave.append(pareto_y_sim[find_cost[j]])
#         ave_prob.append(np.mean(ave))
#         ave_cost.append(costs[i])

# pareto_x_sim, pareto_y_sim = (list(t) for t in zip(*sorted(zip(ave_cost, ave_prob))))

# # pareto_x_f1 = []
# # pareto_y_f1 = []
# # for k in range(len(lst_train)):
# # 	pareto_x_f1.append(lst_train_MO[k])
# # 	pareto_y_f1.append(score_train_MO[k])

# pareto_x_node = []
# pareto_y_node = []
# for k in range(len(class_prob)):
# 	if costs_nodes[k] != 0.0:
# 		indx = find(costs_nodes,costs_nodes[k])
# 		if len(indx) > 10:
# 			max_indx_value = []
# 			for j in range(len(indx)):
# 				max_indx_value.append(class_prob[indx[j]])
# 			max_indx = np.argwhere(max_indx_value == np.max(max_indx_value))
# 			max_indx = random.choice(max_indx)
# 			pareto_x_node.append(costs_nodes[indx[max_indx[0]]])
# 			pareto_y_node.append(class_prob[indx[max_indx[0]]])

# pareto_x_f1_SO = []
# pareto_y_f1_SO = []
# for k in range(len(lst_test_SO)):
# 	pareto_x_f1_SO.append(lst_test_SO[k])
# 	pareto_y_f1_SO.append(score_test_SO[k])

# name = 'lst_test_SO' + str(random_state) + '_1.txt'
# import json
# import ast  

# with open(name, 'r') as f:
#   lst_test_SO = json.loads(f.read())

# name = 'score_test_SO' + str(random_state) + '_1.txt'
# import json
# import ast  

# with open(name, 'r') as f:
#   score_test_SO = json.loads(f.read())

# u = np.diff(pareto_x_sim)
# v = np.diff(pareto_y_sim)
# pos_x = pareto_x_sim[:-1] + u/2
# pos_y = pareto_y_sim[:-1] + v/2
# norm = np.sqrt(u**2+v**2)

# u_f1 = np.diff(pareto_x_f1)
# v_f1 = np.diff(pareto_y_f1)
# pos_x_f1 = pareto_x_f1[:-1] + u_f1/2
# pos_y_f1 = pareto_y_f1[:-1] + v_f1/2
# norm_f1 = np.sqrt(u_f1**2+v_f1**2)

# hv_hv_mean = np.mean(hv_hv_total,axis=0)
# # x = []
# # y = []
# # for j in range(len(P_total)):
# # 	P_total[j].sort()
# # 	x_ind = []
# # 	y_ind = []
# # 	for k in range(len(P_total[j])):
# # 		x_ind.append(P_total[j][k][0])
# # 		y_ind.append(P_total[j][k][1])
# # 	x.append(x_ind)
# # 	y.append(y_ind)

# # x_node = []
# # y_node = []
# # for j in range(len(pareto_x_total)):
# # 	lst = [list(x) for x in zip(costs_nodes[j],class_prob[j])]
# # 	x_node_ind = []
# # 	y_node_ind = []
# # 	for k in range(len(lst)):
# # 		x_node_ind.append(lst[k][0])
# # 		y_node_ind.append(lst[k][1])
# # 	x_node.append(x_node_ind)
# # 	y_node.append(y_node_ind)

#for j in range(len(pareto_x_total)):
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.plot(pareto_x_node,pareto_y_node, marker='o',color='b',label='Node')
ax.plot(pareto_x, pareto_y, marker='o',color='r', label = 'Global Pareto Front')
#ax.plot(pareto_x_sim, pareto_y_sim, marker='o',color='b', label = 'Local Pareto Front')
ax.legend()
ax.set_ylim(0,1.2)
ax.set_xlim(-1.0,0)
ax.set_xlabel('Cost')
ax.set_ylabel('Classification Probability')