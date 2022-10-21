'''
This script takes command line arguments and outputs the MNIST dataset and the respective models

input: [1] = random state
input: [2] = seed value
input: [3] = Number of epochs for CNN training
input: [4] = batch size for CNN training
input: [5] = cnn or lr
input: [6] = Boolean for random strategy (optional)

outputs:

1) text files of mnist data set split based on random state
2) models (cnn or lr) trained on the split (1) with seed value

'''

import itertools
import time

from sklearn.linear_model import LogisticRegression as sk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

import numpy as np
import pandas as pd
import random
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

import sys

random_state = int(sys.argv[1])
seed_value = int(sys.argv[2])
batch_size = int(sys.argv[3])
epochs = int(sys.argv[4])
model_type = sys.argv[5]

try:
	random_strategy = sys.argv[6].lower() == 'true'
	model_name = 'random'
except IndexError:
	random_strategy = 'False'
	model_name = ''

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
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size=train_size,random_state=random_state)

# Set number of categories
num_category = 10

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_val = keras.utils.to_categorical(y_val, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

if random_strategy:
	# Random strategy
	# Train a logistic regression / CNN classifier on random subsets of features,

	#1) Make  

	def make(action,vec,i,x_set):

		tup = tuple(vec)

		for j in range(len(action)):

			val = tuple([x_set[i,action[j]]])

			tup = tup[:action[j]] + val + tup[action[j]+1:]

		return np.asarray(tup)

	print('Creating random feature subset training and test data sets.\n')
	x_train_random = []
	x_val_random = []

	# Split the images into 4x4 blocks with cost of 16 each (1 per pixel)
	action_space = [i for i in range(49)]

	for i in range(len(x_train)):
		print(i)
		tp = (0.0,)*784
		# Set a number of features in each training sample
		action_rand = random.choice(action_space)
		action_lst = random.sample(action_space,action_rand)
		state = np.array(tp,dtype=np.float64)
		state = make(action_lst,state,i,x_train)
		x_train_random.append(state)

	for i in range(len(x_val)):
		print(i)
		tp = (0.0,)*784
		# Set a number of features in each validation sample
		action_rand = random.choice(action_space)
		action_lst = random.sample(action_space,action_rand)
		state = np.array(tp,dtype=np.float64)
		state = make(action_lst,state,i,x_val)
		x_val_random.append(state)

	x_train_random = np.array(x_train_random)
	x_val_random = np.array(x_val_random)

if model_type == 'lr':

	x_train = x_train.reshape((-1,784))
	x_val = x_val.reshape((-1,784))

	print('Training logistic regression\n')

	# Create regularization penalty space
	penalty = ['l1', 'l2','none']

	# Create regularization hyperparameter space
	C = np.logspace(0, 4, 10)

	solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

	iter_num = [100, 1000, 10000, 100000]

	# Create hyperparameter options
	grid_values = dict(C=C, penalty=penalty,solver=solver,max_iter=iter_num)

	model_sk = sk(random_state = seed_value)
	logreg_cv = GridSearchCV(model_sk,grid_values,cv=5)

	if not random_strategy:
		#Pretrain and Retrain
		logreg_cv.fit(x_train, y_train)
		print('Tuned hyperparameters :(best parameters)\n',logreg_cv.best_params_)
		print('Accuracy :\n',logreg_cv.best_score_)
		
		model = sk(C = logreg_cv.best_params_['C'],solver = logreg_cv.best_params_['solver'], max_iter = logreg_cv.best_params_['max_iter'], penalty = logreg_cv.best_params_['penalty'])
		
		model.fit(x_train, y_train)
		predictions = model.predict(x_val)

		print('Confusion matrix of fitted logistic regression: \n',metrics.confusion_matrix(y_test, predictions))
		print('F1 score: \n', metrics.f1_score(y_test,predictions,average='micro'))

		filename = 'finalized_model_mnist_' + str(random_state) + '_' + str(seed_value) + '.sav'
		pickle.dump(model, open(filename, 'wb'))

	else:

		x_train_random = x_train_random.reshape((-1,784))
		x_test_random = x_test_random.reshape((-1,784))

		logreg_cv.fit(x_train_random, y_train)
		print('Tuned hyperparameters :(best parameters)\n',logreg_cv.best_params_)
		print('Accuracy :\n',logreg_cv.best_score_)
		
		model = sk(C = logreg_cv.best_params_['C'],solver = logreg_cv.best_params_['solver'], max_iter = logreg_cv.best_params_['max_iter'], penalty = logreg_cv.best_params_['penalty'])
		
		model.fit(x_train_random, y_train)
		predictions = model.predict(x_test_random)

		print('Confusion matrix of fitted logistic regression: \n',metrics.confusion_matrix(y_test, predictions))
		print('F1 score: \n', metrics.f1_score(y_test,predictions,average='micro'))

		filename = 'finalized_model_mnist_random_' + str(random_state) + '_' + str(seed_value) + '.sav'
		pickle.dump(model, open(filename, 'wb'))

else:

	x_train = x_train.reshape((-1,28,28,1))
	x_val = x_val.reshape((-1,28,28,1))
	x_test = x_test.reshape((-1,28,28,1))

	batch_size = 256

	model = Sequential()

	model.add(Conv2D(filters=64, kernel_size = (3,3), dilation_rate = (2,2), padding = 'same', activation="relu", input_shape=(28,28,1)))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(filters=128, kernel_size = (3,3), dilation_rate = (2,2), padding = 'same', activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(filters=256, kernel_size = (3,3), dilation_rate = (2,2), padding = 'same', activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	    
	model.add(Flatten())

	model.add(Dense(512,activation="relu"))
	    
	model.add(Dense(10,activation="softmax"))

	# Instantiate an optimizer.
	lr = 1e-5

	optimizer = keras.optimizers.Adam(learning_rate=lr)

	# Instantiate a loss function.
	loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
	
	# Prepare the metrics.
	train_acc_metric = keras.metrics.CategoricalAccuracy()
	val_acc_metric = keras.metrics.CategoricalAccuracy()
	test_acc_metric = keras.metrics.CategoricalAccuracy()

	#compilte the model 
	model.compile(optimizer=optimizer, loss=loss_fn, metrics = [train_acc_metric])

	#define manual tranining and inference functions 
	@tf.function
	def train_step(x, y):
		with tf.GradientTape() as tape:
			logits = model(x, training=True)
			loss_value = loss_fn(y, logits)
		grads = tape.gradient(loss_value, model.trainable_weights)
		optimizer.apply_gradients(zip(grads, model.trainable_weights))
		train_acc_metric.update_state(y, logits)
		return loss_value

	@tf.function
	def val_step(x, y):
		val_logits = model(x, training=False)
		loss_value = loss_fn(y,val_logits)
		val_acc_metric.update_state(y, val_logits)
		return loss_value

	@tf.function
	def test_step(x, y):
		test_logits = model(x, training=False)
		loss_value = loss_fn(y,test_logits)
		test_acc_metric.update_state(y, test_logits)
		return loss_value

	loss_train = []
	acc_train = []
	loss_val = []
	acc_val = []

	# Define train function 
	def train(epochs,train_dataset,val_dataset,loss_train,acc_train,loss_val,acc_val,cnn_name):

		print('Training CNN\n')

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

			loss_val_epoch = 0.0
			# Run a validation loop at the end of each epoch.
			for step_val, (x_batch_val, y_batch_val) in enumerate(val_dataset):
				loss_value_val = val_step(x_batch_val, y_batch_val)
				loss_val_epoch += loss_value_val
			loss_val_epoch /= (step_val+1)
			loss_val.append(loss_val_epoch)

			val_acc = val_acc_metric.result()

			test_acc = test_acc_metric.result()

			if not any(i >= val_acc for i in acc_val):
				print('Saving')
				symbolic_weights = getattr(model.optimizer, 'weights')
				weight_values = K.batch_get_value(symbolic_weights)
				opt_name = cnn_name + '_optimizer.pkl'
				with open(opt_name, 'wb') as f:
					pickle.dump(weight_values, f)
				weight_name = cnn_name + '.h5'
				model.save_weights(weight_name)

			acc_val.append(val_acc)

			print("Validation acc: %.4f" % (float(val_acc),))
			print("Validation loss: %.4f" % (float(loss_val_epoch),))

			print("Test acc: %.4f" % (float(test_acc),))

			val_acc_metric.reset_states()
			
			test_acc_metric.reset_states()

			print("Time taken: %.2fs" % (time.time() - start_time))

			return loss_train, loss_val, acc_train, acc_val

	if not random_strategy:

		# Prepare the training dataset.
		train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
		train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

		val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
		val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)

		cnn_name = 'mnist_cnn_' + str(random_state) + '_' + str(seed_value)

		loss_train, loss_val, acc_train, acc_val = train(epochs,train_dataset,val_dataset,loss_train,acc_train,loss_val,acc_val,cnn_name)

	else:

		x_train_random = x_train_random.reshape((-1,28,28,1))
		x_val_random = x_test_random.reshape((-1,28,28,1))

		# Prepare the training dataset.
		train_dataset = tf.data.Dataset.from_tensor_slices((x_train_random, y_train))
		train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

		val_dataset = tf.data.Dataset.from_tensor_slices((x_val_random, y_val))
		val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)

		cnn_name = 'mnist_cnn_random_' + str(random_state) + '_' + str(seed_value)

		loss_train, loss_val, acc_train, acc_val = train(epochs,train_dataset,val_dataset,loss_train,acc_train,loss_val,acc_val,cnn_name)
 
# Output training data to file
print('random_state:\n', random_state)
print('seed_value:\n', seed_value)
print('batch_size:\n', batch_size)
print('epochs:\n', epochs)
print('model_type:\n', model_type)
print('random_strategy:\n', random_strategy)

x_train_name = 'x_train_' + model_name + '_' + str(random_state) + '_' + str(seed_value) + '.txt'
x_val_name = 'x_val_' + model_name + '_' + str(random_state) + '_' + str(seed_value) + '.txt'
x_test_name = 'x_test_' + model_name + '_' + str(random_state) + '_' + str(seed_value) + '.txt'

y_train_name = 'y_train_' + model_name + '_' + str(random_state) + '_' + str(seed_value) + '.txt'
y_val_name = 'y_val_' + model_name + '_' + str(random_state) + '_' + str(seed_value) + '.txt'
y_test_name = 'y_test_' + model_name + '_' + str(random_state) + '_' + str(seed_value) + '.txt'

x_train_file = open(x_train_name, 'w+')
x_val_file = open(x_val_name, 'w+')
x_test_file = open(x_test_name, 'w+')

y_train_file = open(y_train_name, 'w+')
y_val_file = open(y_val_name, 'w+')
y_test_file = open(y_test_name, 'w+')

if random_strategy:
	content = str(x_train_random)
	x_train_file.write(content)
	x_train_file.close()

	content = str(x_val_random)
	x_val_file.write(content)
	x_val_file.close()
else:
	content = str(x_train)
	x_train_file.write(content)
	x_train_file.close()

	content = str(x_val)
	x_val_file.write(content)
	x_val_file.close()

content = str(x_test)
x_test_file.write(content)
x_test_file.close()

content = str(y_train)
y_train_file.write(content)
y_train_file.close()

content = str(y_val)
y_val_file.write(content)
y_val_file.close()

content = str(y_test)
y_test_file.write(content)
y_test_file.close()

