'''

Split the MNIST data set into train and test with input random seeds

Train a CNN 

'''


import pandas as pd
import numpy as np
import itertools
import time
from sklearn.linear_model import LogisticRegression as sk
from sklearn.model_selection import train_test_split
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

random_state = 1
seed_value = 12321

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

x_train = x_train.reshape((-1,28,28,1))
x_val = x_val.reshape((-1,28,28,1))
x_test = x_test.reshape((-1,28,28,1))

# Set number of categories
num_category = 10

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_val = keras.utils.to_categorical(y_val, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

# Define the CNN architecture
tf.random.set_seed(seed_value)

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
epochs = 71
#decay_rate = lr/epochs
optimizer = keras.optimizers.Adam(learning_rate=lr)

# Instantiate a loss function.
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

''' 

Train a CNN on random subsets of features,

1) Make  

def make(action,vec,i,x_set):

    tup = tuple(vec)

    for j in range(len(action)):

        val = tuple([x_set.iloc[i,action[j]]])

        tup = tup[:action[j]] + val + tup[action[j]+1:]

    return np.asarray(tup)

x_train_new = []
x_val_new = []

# Split the images into 4x4 blocks with cost of 16 each (1 per pixel)
action_space = [i for i in range(49)]

m,n = x_train.shape
for i in range(m):
    print(i)
    tp = (0.0,)*784
    # Set a number of features in each training sample
    action_rand = random.choice(action_space)
    action_lst = random.sample(action_space,action_rand)
    state = np.array(tp,dtype=np.float64)
    state = make(action_lst,state,i,x_train)
    x_train_new.append(state)

m,n = x_val.shape
for i in range(m):
    print(i)
    tp = (0.0,)*784
    # Set a number of features in each validation sample
    action_rand = random.choice(action_space)
    action_lst = random.sample(action_space,action_rand)
    state = np.array(tp,dtype=np.float64)
    state = make(action_lst,state,i,x_val)
    x_train_new.append(state)

x_train_new = np.array(x_train_new)
x_val_new = np.array(x_test_new)

x_train_new = x_train_new.reshape((-1,28,28,1))
x_val_new = x_val_new.reshape((-1,28,28,1))

'''

# Prepare the training dataset.
batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the metrics.
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()
test_acc_metric = keras.metrics.CategoricalAccuracy()

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(batch_size)

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

loss_train_model = 0.0
for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
    #define the loss based on initial weights
    check = np.ones((y_batch_train.shape[0],784))
    check = np.reshape(check,(-1,28,28,1))
    loss_value = test_step(check, y_batch_train)
    loss_train_model += loss_value
loss_train_model /= (step+1)

# Display metrics at the end of each epoch.
train_acc_model = train_acc_metric.result()

loss_val_model = 0.0
for step_val, (x_batch_val, y_batch_val) in enumerate(val_dataset):
    #define the loss based on initial weights
    check = np.zeros((y_batch_val.shape[0],784))
    check = np.reshape(check,(-1,28,28,1))
    loss_value = test_step(check, y_batch_val)
    loss_val_model += loss_value
loss_val_model /= (step_val+1)

# Display metrics at the end of each epoch.
val_acc_model = val_acc_metric.result()

import time

loss_train = [loss_train_model]
acc_train = [train_acc_model]
loss_val = [loss_val_model]
acc_val = [val_acc_model]

model_name = 'mnist_cnn_' + str(random_state) + '_' + str(seed_value)

def train(epochs,train_dataset,val_dataset,loss_train,acc_train,loss_val,acc_val,model_name):

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
			opt_name = model_name + '_optimizer.pkl'
			with open(opt_name, 'wb') as f:
				pickle.dump(weight_values, f)
			weight_name = model_name + '.h5'
			model.save_weights(weight_name)

		acc_val.append(val_acc)

		print("Validation acc: %.4f" % (float(val_acc),))
		print("Validation loss: %.4f" % (float(loss_val_epoch),))

		print("Test acc: %.4f" % (float(test_acc),))

		val_acc_metric.reset_states()
		
		test_acc_metric.reset_states()

		print("Time taken: %.2fs" % (time.time() - start_time))

	return loss_train, loss_val, acc_train, acc_val

loss_train, loss_val, acc_train, acc_val = train(epochs,train_dataset,val_dataset,loss_train,acc_train,loss_val,acc_val,model_name)
