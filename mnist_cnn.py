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
#from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Sequential
#import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.regularizers import l2
#from tensorflow.keras.regularizers import l1_l2

random_state = 1
seed_value = 12321

#Set a seed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.RandomState(seed_value)

#Setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0

x = np.concatenate((x_train,x_test))
y = np.concatenate((y_train,y_test))

train_size = 0.8

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=train_size,random_state=random_state)

#x_train = x_train.reshape((-1,28,28,1))
#x_test = x_test.reshape((-1,28,28,1))

x_train = x_train.reshape((-1,784))
x_test = x_test.reshape((-1,784))

x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)

#set number of categories
num_category = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

tf.random.set_seed(seed_value)

model = Sequential()

#model.add(Lambda(standardize,input_shape=(28,28,1)))    
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
    
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

def num_features(state):

    lst = [[i] for i in range(0,784)]

    pairs = [[i,i] for i in range(0,784)]

    dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

    acquired = []
    for k in range(len(lst)):
        lst_to_be_checked = lst[k]
        #if lst_to_be_checked[0] <= 33 and tup[lst_to_be_checked[0]] != 1:
        #    acquired.append(lst_to_be_checked[0])
        if state[lst_to_be_checked[0]] != -1:
            acquired.append(lst_to_be_checked[0])

    dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

    num_features = list(set(dct_indx))

    return num_features

def make_train(action,vec,i):

    tup = tuple(vec)

    for j in range(len(action)):

        val = tuple([x_train.iloc[i,action[j]]])

        tup = tup[:action[j]] + val + tup[action[j]+1:]

    return np.asarray(tup)

def make_test(action,vec,i):

    tup = tuple(vec)

    for j in range(len(action)):

        val = tuple([x_test.iloc[i,action[j]]])

        tup = tup[:action[j]] + val + tup[action[j]+1:]

    return np.asarray(tup)


def cost(state):

    cost = 0

    lst_features = num_features(state)

    for i in range(len(lst_features)):
        cost += 1

    return cost

def get_mask(s):

    lst = [i for i in range(0,784)]

    #pairs = [[i,i] for i in range(0,784)]

    #dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

    full_features = [i for i in range(0,784)]

    to_be_acquired = [lst[i] for i in range(len(lst)) if s[lst[i]] == -1.0]

    mask = [0 if full_features[k] in to_be_acquired else 1 for k in range(len(full_features))]

    mask = np.array(mask,dtype='f')

    return mask

x_train_new = []
x_test_new = []

total_cost = 784.0

#coeff = 0

m,n = x_train.shape
for i in range(m):
    print(i)
    cost_rand = random.choice(np.arange(785))
    lst_rand = random.sample([i for i in range(0,784)],cost_rand)
    tp = (1,)*784
    state = np.array(tp,dtype=np.float64)
    state = make_train(lst_rand,state,i)
    #mask_state = get_mask(state)
    #state = np.concatenate((state,mask_state))
    #state = np.array([quad_end(total_cost,coeff,cost_rand) if state[i] == -1.0 else state[i] for i in range(len(state))])
    x_train_new.append(state)
    #print(cost(state))

m,n = x_test.shape
for i in range(m):
    print(i)
    cost_rand = random.choice(np.arange(785))
    lst_rand = random.sample([i for i in range(0,784)],cost_rand)
    tp = (-1,)*784
    state = np.array(tp,dtype=np.float64)
    state = make_test(lst_rand,state,i)
    #mask = get_mask(state)
    #state = np.concatenate((state,mask))
    x_test_new.append(state)
    #print(cost(state))

x_train_new = np.array(x_train_new)
x_test_new = np.array(x_test_new)

x_train_new = x_train_new.reshape((-1,28,28,1))
x_test_new = x_test_new.reshape((-1,28,28,1))

# Prepare the training dataset.
batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_new, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the metrics.
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()
test_acc_metric = keras.metrics.CategoricalAccuracy()

val_dataset = tf.data.Dataset.from_tensor_slices((x_test_new, y_test))
val_dataset = val_dataset.batch(batch_size)

#test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
#test_dataset = test_dataset.batch(batch_size)

model.compile(optimizer=optimizer, loss=loss_fn, metrics = [train_acc_metric])

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

@tf.function
def test_step(x, y):
    test_logits = model(x, training=False)
    #test_logits = tf.reshape(test_logits,(-1,10))
    loss_value = loss_fn(y,test_logits)
    test_acc_metric.update_state(y, test_logits)
    return loss_value

loss_train_model = 0.0
for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
    check = np.ones((y_batch_train.shape[0],784))
    check = np.reshape(check,(-1,28,28,1))
    loss_value = test_step(check, y_batch_train)
    loss_train_model += loss_value
loss_train_model /= (step+1)

# Display metrics at the end of each epoch.
train_acc_model = val_acc_metric.result()

loss_test_model = 0.0
for step_val, (x_batch_val, y_batch_val) in enumerate(val_dataset):
    check = np.ones((y_batch_val.shape[0],784))
    check = np.reshape(check,(-1,28,28,1))
    loss_value = test_step(check, y_batch_val)
    loss_test_model += loss_value
loss_test_model /= (step_val+1)

# Display metrics at the end of each epoch.
val_acc_model = val_acc_metric.result()

import time

loss_train = [loss_train_model]
acc_train = [train_acc_model]
loss_val = [loss_test_model]
acc_val = [val_acc_model]

epochs = 101
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

    #test_acc = test_acc_metric.result()

    #if not any(i >= val_acc for i in acc_val):
    #    print('Saving')
    #    model_name = "mnist_cnn_random_" + str(random_state) + '_' + str(seed_value)# + ".h5"
    #    symbolic_weights = getattr(model.optimizer, 'weights')
    #    weight_values = K.batch_get_value(symbolic_weights)
    #    opt_name = model_name + '_optimizer.pkl'
    #    with open(opt_name, 'wb') as f:
    #        pickle.dump(weight_values, f)
    #    weight_name = model_name + '.h5'
    #    model.save_weights(weight_name)

    acc_val.append(val_acc)

    print("Validation acc: %.4f" % (float(val_acc),))
    print("Validation loss: %.4f" % (float(loss_val_epoch),))

    #print("Test acc: %.4f" % (float(test_acc),))

    val_acc_metric.reset_states()
    
    #test_acc_metric.reset_states()

    print("Time taken: %.2fs" % (time.time() - start_time))
