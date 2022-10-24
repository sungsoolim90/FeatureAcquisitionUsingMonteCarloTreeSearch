'''
This script implements the Deep Q-Network algorith on tensorflow
for the feature acquisition problem of MNIST. 

a) Each 4x4 block of pixels is defined as a feature with cost of 16
b) Feature acquisition episode starts from the empty feature state and ends in the full feature state

input: [1] = random state
input: [2] = seed value
input: [3] = Number of epochs for DQN training
input: [4] = lr or cnn for classifier
input: [5] = Boolean for random strategy (optional)

outputs:

1) text files of loss and cumulative rewards at end of training
2) DQN policy weights at end of each epoch
3) Inference samples

'''

import pickle 
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2

import itertools
import time
import json
import heapq
from collections import deque

import argparse

parser = argparse.ArgumentParser(
    description='''Here are the arguments to train_classifiers.py. ''',
    epilog="""Outputs \n
    1) text files of loss and cumulative rewards at end of training \n
    2) DQN policy weights at end of each epoch\n
    3) Inference samples.""")
parser.add_argument('--rs', required =True, type=int, default=1, help='random state (required)')
parser.add_argument('--sv', required = True, type=int, default=12321, help='seed value (reqired)')
parser.add_argument('--mt', required = True, type=str, default='lr', help='model type: lr or cnn (required)')
parser.add_argument('--e', required = True, type=int, default=10, help='number of epochs for PPO training (required)')
parser.add_argument('--st', required = False, type=bool, help='random strategy: True or False (optional)')

args = parser.parse_args()

random_state = args.rs
seed_value = args.sv
epochs = args.e
model_type = args.mt
random_strategy = args.st

if random_strategy:
    model_name = 'random'
else:
    model_name = ''

retrain = 60000000 # no CNN retraining for pretrain and random
purge_step = 60000000

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

#Set seed for tensorflow
tf.random.set_seed(seed_value)

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

'''
class DQNAgent: model and target model with the replay memory buffer

'''

class DQNAgent:
    def __init__(self, state_size, action_size, n_latent_var):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.5
        self.learning_rate = 0.0000001
        self.model = self._build_model(n_latent_var)
        self.target_model = self._build_model(n_latent_var)
        self.update_target_model()

    def reset_epsilon(self):
        self.epsilon = 1.0

    def predict(self,X):
        W = self.model.get_weights()
        X      = X.reshape((X.shape[0],-1))           #Flatten
        X      = X @ W[0] + W[1]                      #Dense
        X[X<0] = 0                                    #Relu
        X      = X @ W[2] + W[3]                      #Dense
        X[X<0] = 0                                    #Relu
        X      = X @ W[4] + W[5]                      #Dense
        X[X<0] = 0                                    #Relu
        X      = X @ W[6] + W[7]                      #Dense
        X      = np.exp(X)/np.exp(X).sum(1)[...,None] #Softmax
        return X

    def predict_target(self,X):
        W = self.target_model.get_weights()
        X      = X.reshape((X.shape[0],-1))           #Flatten
        X      = X @ W[0] + W[1]                      #Dense
        X[X<0] = 0                                    #Relu
        X      = X @ W[2] + W[3]                      #Dense
        X[X<0] = 0                                    #Relu
        X      = X @ W[4] + W[5]                      #Dense
        X[X<0] = 0                                    #Relu
        X      = X @ W[6] + W[7]                      #Dense
        X      = np.exp(X)/np.exp(X).sum(1)[...,None] #Softmax
        return X

    """Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self, n_latent_var):
        # Deep-Q Network
        initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal')
        model = Sequential()
        model.add(Dense(n_latent_var, input_dim=self.state_size, activation='relu', kernel_initializer=initializer))
        model.add(Dense(n_latent_var/2, activation='relu',kernel_initializer=initializer))
        model.add(Dense(n_latent_var/4, activation='relu',kernel_initializer=initializer))
        #model.add(Dense(128, activation='relu',kernel_initializer=initializer))
        model.add(Dense(self.action_size, activation='linear',kernel_initializer=initializer))
        model.compile(loss=self._huber_loss,optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        #weights = self.model.get_weights()
        #target_weights = self.target_model.get_weights()
        #set_weights = tf.multiply(0.999,target_weights) + tf.multiply(0.001,weights)
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def clear_replay(self):
        self.memory.clear()

    def act(self, state):
        # epsilon-decay stochastic policy
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(range(self.action_size), 1)
            act_values = np.zeros((1,784))
            act_values = act_values[0]
            act_values[action] = 1
            return act_values
        act_values = self.predict(state)
        return act_values[0]#returns action probabilities

    def infer(self, state):
        act_values = self.predict(state)
        return act_values[0]#returns action probabilities

    def train_step(self, state, target):
        with tf.GradientTape() as tape:
            logits = self.model(state, training=True)
            loss_value = self._huber_loss(target,logits,clip_delta=1.0)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value

    def replay(self, batch_size):
        # Update network using replay buffer
        minibatch = random.sample(self.memory, batch_size)

        reward_list = []
        for k in range(len(minibatch)):
            reward_list.append(minibatch[k][2])

        mean_reward = np.mean(reward_list)
        std_reward = np.std(reward_list)

        for j in range(len(minibatch)):
            minibatch[j] = np.array(minibatch[j],dtype=object)
            minibatch[j][2] = (minibatch[j][2] - mean_reward)/std_reward
            minibatch[j] = tuple(minibatch[j])

        total_loss = 0.0

        v = 1
        for state, action, reward, next_state, done in minibatch:
            target = self.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.predict(next_state)[0]
                t = self.predict_target(next_state)[0] 
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            loss_value = self.train_step(state, target)
            total_loss += loss_value
            v += 1
        
        total_loss /= v

        if self.epsilon > self.epsilon_min: #every time it gets updated
            self.epsilon *= self.epsilon_decay

        return total_loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def reward(state,total_cost,model_class,action_lst):

    cost = 0.0

    state = np.asarray(state).flatten()

    for i in range(len(action_lst)):
        cost += 16

    try:
        state = state.reshape((-1,784))
        classification_prob = np.max(model_class.predict_proba(state).flatten())
    except AttributeError:
        state = state.reshape((-1,28,28,1))
        classification_prob = np.max(model_class.predict(state)).flatten()

    cost = (cost + 1.0)/(total_cost + 1.0) #cost always increasing from 0 to 1
    return_value = classification_prob/cost
    return_value = return_value.flatten()

    return return_value

def make(action,state,i,action_lst,x_set):

    state = np.asarray(state).flatten()

    state = state.reshape((-1,28,28,1))

    row_action = int(action/7.0)

    column_action = int(action - row_action*7.0)

    state[:,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1)] = x_set[i,column_action*4:(column_action+1)*4,4*row_action:4*(row_action+1),:]

    state = state.reshape((-1,784))

    is_terminal = (len(action_lst)+1) == 49

    return state[0], is_terminal #new node

# logging variables
state_size = 784
action_size = 49
n_latent_var = 512
agent = DQNAgent(state_size, action_size, n_latent_var)

############## Hyperparameters ##########################################
batch_size = int((action_size+1)/2)      # batch_size for DQN update
update_timestep = int((action_size+1)/2) # update frequency for DQN
update_target = int((action_size+1))     # update frequency for DQN target
total_cost = 784.0                       # total cost of full features
max_episodes = 100                       # number of episodes 
max_timesteps = action_size              # time steps in an episode
#########################################################################

loss_total = []
reward_total = []

m,n,o,p = x_train.shape

# Training loop
for j in range(num_epochs):
    
    print(j)

    reward_per_epoch = []
    loss_per_epoch =[]

    # Define copies for retraining classifier
    
    # X_train_z_copy = x_train.copy()
    # y_train_copy = y_train.copy()

    # Define variables for retraining classifier
    
    # purge_time = 1
    # retrain_time = 1
    # retrain_node = []


    for i in range(m):
        
        print(i)

        loss_per_sample = []
        reward_per_sample = []

        timestep_replay = 0
        timestep_target = 0

        #retrain_indx = random.choice(range(action_size))

        for e in range(max_episodes):

            act = []
            tp = (0.0,)*784

            state = np.array(tp,dtype=np.float64)
            state = np.reshape(state, [1, state_size])

            reward_per_episode = []
            loss_per_episode = []

            for time_step in range(action_size):

                timestep_replay += 1
                timestep_target += 1

                action_values = agent.act(state)

                action = heapq.nlargest(action_size,range(len(action_values)), key=action_values.__getitem__)

                st =  set(act)

                to_be = [ele for ele in action if ele not in st]

                action = to_be[0]

                next_state, done = make(action,state,i,act,x_train)

                reward_state = reward(next_state,total_cost,model_class,act)

                act.append(action)

                #if e == (EPISODES - 1):

                #    if time_step == retrain_indx:

                #        retrain_node.append(next_state)

                next_state = np.reshape(next_state, [1, state_size])
                agent.memorize(state, action, reward_state, next_state, done)
                state = next_state

                reward_per_episode.append(reward_state)#,mask))

                if done:
                    break

                if timestep_target % update_target == 0:

                    agent.update_target_model()

                    timestep_target = 0

                if timestep_replay % update_timestep == 0:

                    loss = agent.replay(batch_size)
                    
                    loss_per_episode.append(loss)

                    timestep_replay = 0

            #Reset epsilon at end of an episode
            agent.reset_epsilon()

            reward_per_sample.append(np.sum(reward_per_episode))
            loss_per_sample.append(np.sum(loss_per_episode))

# For the retrain strategy, where LR or CNN classifier is periodically retrained
# on the states visited during the training of the PPO algorithm

        #if (i+1) % retrain == 0:
            
        #    print('Retraining CNN')

        #    indices = []
        #    for q in range(len(retrain_node_transform)):
        #        z = random.choice(range(len(retrain_node_transform)))
        #        indices.append(z) #1 or 2 

        #    retrain_node_transform_selected = [retrain_node_transform[indices[z]] for z in range(len(indices))]

        #    retrain_node_transform_selected = pd.DataFrame(retrain_node_transform_selected)

        #    X_train_z_new = pd.concat([X_train_z_copy,retrain_node_transform_selected],ignore_index = True)

        #    y_train_new_z = []
        #    for k in range(len(retrain_node)):
        #        if purge_time > 1:
        #            ind = int((indices[k]+purge_step*purge_time)/(max_episodes*max_timesteps))
        #        else:
        #            ind = int((indices[k])/(max_episodes*max_timesteps))
        #        y_train_new_z.append(y_train_new.iloc[ind])

        #    y_train_new_z = pd.DataFrame(y_train_new_z)

        #    y_train_new_z = pd.concat([pd.DataFrame(y_train_copy),y_train_new_z],ignore_index=True)

        #    if classifier_name == 'lr':
        #        model_class.fit(X_train_new_z,y_train_new_z)
        #    else:
        #        batch_size = 512
        #        train_X_total = np.reshape(X_train_z_new, (-1,28,28,1))
        #        test_X_total = np.reshape(X_test_z, (-1,28,28,1))

        #        train_dataset = tf.data.Dataset.from_tensor_slices((train_X_total, y_train_new_z))
        #        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        #        val_dataset = tf.data.Dataset.from_tensor_slices((X_test_z, y_test))
        #        val_dataset = val_dataset.batch(batch_size)

        #        loss_train, loss_test, acc_train, acc_test = train(train_dataset,val_dataset)

        #if (i + 1) % purge_step == 0:

        #    print('Purging')

        #    q,r = X_train_z_new.shape
        #    m,n = x_train.shape
        #    purge = q - m

        #    indx = random.sample(range(0, q), purge)

        #    indx_name = []

        #    for t in range(len(indx)):
        #        indx_name.append(X_train_z_new.iloc[indx[t],:].name)

        #    X_train_drop = X_train_z_new.drop(index=indx_name)

        #    y_train_drop = y_train_new_z.drop(index=indx_name)

        #    retrain_node = []

        #    X_train_z_copy = X_train_drop

        #    y_train_copy = y_train_drop.to_numpy()

        #    purge_time += 1
                
        loss_per_epoch.append(np.mean(loss_per_sample))
        reward_per_epoch.append(np.mean(reward_per_sample))
        agent.reset_epsilon()
    loss_total.append(np.mean(loss_per_epoch))
    reward_total.append(np.mean(reward_per_epoch))
    print(loss_total)
    print(reward_total)

    # Save loss, reward, model weights
    if (j+1) % num_epochs == 0:

        loss_name = 'loss_dqn_' + classifier_name + '_mnist_' + str(random_state) + '_' + str(j) + '_' + str(retrain) + '.txt'
        with open(val_name, 'w') as f:
            f.write(json.dumps(str(loss_total)))

        reward_name = 'reward_dqn_' + classifier_name + '_mnist_' + str(random_state) + '_' + str(j) + '_' + str(retrain) + '.txt'
        with open(reward_name, 'w') as f:
            f.write(json.dumps(str(reward_total)))

    name = 'DQN_model_weights_' + model_name + '_' + classifier_name + '_mnist_' + str(j) + '_' + str(random_state) +  '_' + str(seed_value) + '_' + str(retrain) + '.h5'
    agent.save(name)
    # Reset epsilon at end of each training sample 
    agent.reset_epsilon()

#Inference step
m,n,o,p = x_test.shape
state_total = []
action_total = []
for i in range(m):
    print(i)
    tp = (0.0,)*784
    state = np.array(tp,dtype=np.float64)        
    state = np.reshape(state,[1,state_size])
    act = []
    act_sample = []
    state_sample = []
    done = False
    while True:

        action_values = agent.infer(state)

        action = heapq.nlargest(action_size,range(len(action_values)), key=action_values.__getitem__)

        st =  set(act)

        to_be = [ele for ele in action if ele not in st]

        action = to_be[0]

        next_state, done = make(action,state,i,act,x_test) #make a child

        act.append(action)

        state_sample.append(state.copy())

        next_state = np.reshape(next_state, [1, state_size])

        state = next_state

        if done:

            state_sample.append(next_state[0].copy())

            act_sample.append(act.copy())

            break

    action_total.append(act_sample)

    state_total.append(state_sample)

state_name = 'node_test_dqn_' + classifier_name + '_mnist_' + str(random_state) + '_' + str(seed_value) + '_' + str(num_epochs) + '_' + str(retrain) + '.txt'
with open(state_name, 'w') as f:
    f.write(json.dumps(str(state_total)))

action_name = 'feature_acquisition_action_sequence_test_dqn_' + classifier_name + '_mnist_' + str(random_state) + '_' + str(seed_value) + '_' + str(num_epochs) + '_' + str(retrain) + '.txt'
with open(action_name, 'w') as f:
    f.write(json.dumps(str(action_total)))



