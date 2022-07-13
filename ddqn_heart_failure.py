
import pandas as pd
import numpy as np
import itertools
import time
from sklearn.linear_model import LogisticRegression as sk
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2

from collections import Counter, deque
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

seed_value = 12321

np.random.RandomState(seed_value)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

random_state = 1

#random.seed(2)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu,True)

#First set tp as realistic as possible based on the data set - needed for logistic regression
df = pd.read_csv('heart_failure.csv')

#Event   Gender  Smoking Diabetes  BP  Anaemia Age Ejection.Fraction   Sodium  Creatinine  Pletelets   CPK
lst1 = ['Gender','Smoking', 'Diabetes', 'BP', 'Anaemia','Age','Ejection.Fraction', 'Sodium', 'Creatinine', 'Pletelets', 'CPK']

#1) label encode 
lst1 = ['Age']
for i in range(0,len(lst1)):
    df[lst1[i]] = df[lst1[i]]//(i*10+10)

new_df = df.dropna() #set Nan values to unphysical values
new_df = new_df.reset_index(drop=True)
    
y = new_df['Event']

counter = Counter(y)
print(counter)

# define pipeline
over = SMOTENC(categorical_features=[2,3,4,5,6,7],sampling_strategy=1.0,random_state = random_state)
under = RandomUnderSampler(sampling_strategy=1.0,random_state = random_state)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
# transform the dataset
X, y = pipeline.fit_resample(np.asarray(new_df), np.asarray(y))
# summarize the new class distribution

lst1 = [2,3,4,5,6,7,8,9,10,11,12]
df2 = pd.DataFrame([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]], columns=lst1)

X = pd.DataFrame(X)
X = X.append(df2)

enc = OneHotEncoder(sparse = False)
scaler = StandardScaler()

#Z-transform the real-valued data
X_transform = pd.DataFrame()
lst2 = [2,3,4,5,6,7]
for i in range(len(lst2)):
    X_con = pd.DataFrame(enc.fit_transform(np.asarray(X.iloc[:,lst2[i]]).reshape(-1,1)))
    X_transform = pd.concat([X_transform,X_con],axis=1)

X_transform = X_transform.iloc[:-1,:]
X = X.iloc[:-1,:]

#0 to 24
lst3 = [8,9,10,11,12]
for i in range(len(lst3)):
    #X = pd.DataFrame(scaler.fit_transform(new_df[[lst3[i]]]))
    X_con = X.iloc[:,lst3[i]]
    X_transform = pd.concat([X_transform,X_con],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_transform, y, test_size=0.2, random_state = random_state)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
#y_train = pd.DataFrame(y_train)
#y_test = pd.DataFrame(y_test)

m,n = X_train.shape

#Z-transform the real-valued data
X_train_z = pd.DataFrame()
lst2 = [i for i in range(0,22)]
for i in range(len(lst2)):
    X = pd.DataFrame(X_train.iloc[:,lst2[i]])
    X_train_z = pd.concat([X_train_z,X],axis=1)

#0 to 24
lst3 = [i for i in range(22,27)]
for i in range(len(lst3)):
    X = pd.DataFrame(scaler.fit_transform(np.asarray(X_train.iloc[:,lst3[i]]).reshape(-1,1)),index=X_train.iloc[:,lst3[i]].index)
    X_train_z = pd.concat([X_train_z,X],axis=1)

X_test_z = pd.DataFrame()
lst2 = [i for i in range(0,22)]
for i in range(len(lst2)):
    X = pd.DataFrame(X_test.iloc[:,lst2[i]])
    X_test_z = pd.concat([X_test_z,X],axis=1)

lst2 = [i for i in range(22,27)]
for i in range(len(lst2)):
    X = pd.DataFrame((X_test.iloc[:,lst2[i]] - np.mean(X_train.iloc[:,lst2[i]]))/np.std(X_train.iloc[:,lst2[i]]))
    X_test_z = pd.concat([X_test_z,X],axis=1)

EPISODES = 100

coeff = -50

# update = 11 (best loss, reward also decreases)
# reset epsilon after each episode (loss decreasing, reward stable)
# e-greedy decay rate (0.997 and 0.9)

#filename = 'finalized_model_hf_smote_random_' + str(random_state) + '_' + str(coeff) + '_linear.sav'
#filename = 'finalized_model_hf_smote_random_' + str(random_state) +  '_' + str(coeff) + '_' + str(seed_value) + '_quadratic_end.sav'

filename = 'finalized_model_hf_smote' + str(random_state) + '_' + str(seed_value) + '.sav'
#filename = 'finalized_model_hf_smote' + str(random_state) + '.sav'
model_sk = pickle.load(open(filename, 'rb'))

#filename = 'finalized_model_hf_smote_fit_' + str(random_state) + '_' + str(seed_value) + '.sav'
#model_fit = pickle.load(open(filename, 'rb'))

def quad_beg(total_cost,coeff,cost):
    a = (coeff) / (total_cost) ** 2 # c = value at high cost, d = high cost (41), k, h=0
    y = a * (cost) ** 2
    return y

def linear_function(c,d,h,k,cost): #c = coeff, d = high cost, h = 0, k = 0
    a = 1
    b = h - a*k
    a = (c - b) / d
    y = a*cost + b
    return y

def quad_end(total_cost,coeff,cost):
    a = -coeff / total_cost ** 2#-y_1/x_1 **2
    b = 2 * coeff / total_cost #2*y_1/x_1
    y = a * cost ** 2 + b * cost 
    return y

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.000001
        self.model = self._build_model()
        self.target_model = self._build_model()
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
        #X      = np.exp(X)/np.exp(X).sum(1)[...,None] #Softmax
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
        #X      = np.exp(X)/np.exp(X).sum(1)[...,None] #Softmax
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

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal')
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu', kernel_initializer=initializer))
        model.add(Dense(16, activation='relu',kernel_initializer=initializer))
        model.add(Dense(8, activation='relu',kernel_initializer=initializer))
        model.add(Dense(self.action_size, activation='linear',kernel_initializer=initializer))
        model.compile(loss=self._huber_loss,optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        #set_weights = tf.multiply(0.999,target_weights) + tf.multiply(0.001,weights)
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def clear_replay(self):
        self.memory.clear()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(range(self.action_size), self.action_size) #random.randrange(self.action_size) #
            act_values = np.zeros((1,11))
            act_values = act_values[0]
            act_values[action] = 1
            return act_values
        act_values = self.predict(state)#model.predict(state,batch_size=1)
        return act_values[0]#np.argmax(act_values[0])  # returns action

    def train_step(self, state, target):
        with tf.GradientTape() as tape:
            logits = self.model(state, training=True)
            loss_value = self._huber_loss(target,logits,clip_delta=1.0)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value

    def replay(self, batch_size):

        reward_list = []
        for k in range(len(self.memory)):
            reward_list.append(self.memory[k][2])

        mean_reward = np.mean(reward_list)
        std_reward = np.std(reward_list)

        for j in range(len(self.memory)):
            self.memory[j] = np.array(self.memory[j],dtype=object)
            self.memory[j][2] = (self.memory[j][2] - mean_reward)/std_reward
            self.memory[j] = tuple(self.memory[j])
            #tp = tuple(tp)
            #new_tp = self.memory[j][:2] + tp + self.memory[j][3:] #self.memory[j][2:3]

        minibatch = self.memory#np.random.choice(self.memory, batch_size)
        #t0 = time.time()
        total_loss = 0.0
        i = 1
        for state, action, reward, next_state, done in minibatch:
            target = self.predict(state)#model.predict(state, batch_size = 1)
            if done:
                target[0][action] = reward
            else:
                a = self.predict(next_state)[0]
                t = self.predict_target(next_state)[0] #target_model.predict(next_state, batch_size = 1)[0]
                #target[0][action] = reward + self.gamma * np.amax(t)
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            loss_value = self.train_step(state, target)
            total_loss += loss_value
            i += 1
        total_loss /= i

        if self.epsilon > self.epsilon_min: #every time it gets updated
            self.epsilon *= self.epsilon_decay

        return total_loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def get_mask(s):

    lst = [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14], [15,16,17,18,19,20,21],[22], [23], [24], [25],[26]]

    dict_lst = {0: [0,1,2], 3: [3,4,5], 6: [6,7,8], 9: [9,10,11], 12: [12,13,14], 15: [15,16,17,18,19,20,21], 22: [22], 23: [23], 24: [24], 25: [25], 26: [26]}

    full_features = [0, 3, 6, 9, 12, 15, 22, 23, 24, 25,26]

    to_be_acquired = []
    for k in range(len(lst)):
        lst_to_be_checked = lst[k]
        if lst_to_be_checked[0] <= 21 and s[lst_to_be_checked[0]] == 1:
            to_be_acquired.append(lst_to_be_checked[0])
        elif lst_to_be_checked[0] > 21 and s[lst_to_be_checked[0]] == -1:
            to_be_acquired.append(lst_to_be_checked[0])

    mask = np.zeros((1,27))
    mask = mask[0]
    for i in range(len(full_features)):
        if full_features[i] in to_be_acquired:
            mask_to_be_filled = dict_lst[full_features[i]]
            for k in range(len(mask_to_be_filled)):
                mask[mask_to_be_filled[k]] = 0
        else:
            mask_to_be_filled = dict_lst[full_features[i]]
            for z in range(len(mask_to_be_filled)):
                mask[mask_to_be_filled[z]] = 1

    mask = np.array(mask,dtype='f')
    return mask

def reward(state):#,mask):

    cost = 0
    total_cost = 41

    lst = [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14], [15,16,17,18,19,20,21], [22], [23], [24], [25], [26]]
    
    dict_lst = {0: [0,1,2], 1: [3,4,5], 2: [6,7,8], 3: [9,10,11], 4: [12,13,14], 5: [15,16,17,18,19,20,21], 6: [22], 7: [23], 8: [24], 9: [25], 10: [26]}

    acquired = []
    for i in range(len(lst)):
        lst_to_be_checked = lst[i]
        if lst_to_be_checked[0] <= 21 and state[lst_to_be_checked[0]] != 1:
            acquired.append(lst_to_be_checked[0])
        elif lst_to_be_checked[0] > 21 and state[lst_to_be_checked[0]] != -1:
            acquired.append(lst_to_be_checked[0])

    dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

    num_features = list(set(dct_indx))

    for i in range(len(num_features)):
        if num_features[i] <= 5:
            cost += 1
        else:
            cost += 7

    state = np.asarray(state).flatten()

    #mask = get_mask(state)

    indx = [dict_lst[feature] for feature in dct_indx]
    flat_list = [item for sublist in indx for item in sublist]

    #if not flat_list:
        #state = state.reshape(1,-1)
    model = model_sk
    #else:
        #filename = str(flat_list) + '.h5'
        #model = tf.keras.models.load_model(filename)
    #    state = np.array([val for i,val in enumerate(state) for j,index in enumerate(flat_list) if i == index])
        #mask = np.ones((1,len(state)))
        #mask = mask[0]
        #mask = np.array(mask,dtype='f')
        #state = np.concatenate((state,mask))
        #state = state.reshape(1,-1)
    #    model = model_fit[tuple(flat_list)]

    for i in range(len(state)):
        if state[i] == -1:
            state[i] = quad_end(total_cost,coeff,cost)#coeff#linear_function(coeff,total_cost,0,0,cost)

    # X = state
    # W = model.get_weights()
    # X      = X.reshape((X.shape[0],-1))           #Flatten
    # X      = X @ W[0] + W[1]                      #Dense
    # X[X<0] = 0                                    #Relu
    # X      = X @ W[2] + W[3]                      #Dense
    # X[X<0] = 0                                    #Relu
    # X      = X @ W[4] + W[5]                      #Dense
    # X[X<0] = 0                                    #Relu
    # X      = np.exp(X)/np.exp(X).sum(1)[...,None] #Softmax
    
    #classification = np.max(X)
    state = state.reshape(1,-1)
    classification = model.predict(state) #0 or 1
    prob = model.predict_proba(state).flatten()
    classification = prob[int(classification[0])]
    cost = (cost + 1.0)/(total_cost + 1.0) #cost always increasing from 0 to 1
    return_value = classification/cost
    return_value = return_value.flatten()

    return return_value

def num_features(state):

    lst = [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14], [15,16,17,18,19,20,21], [22], [23], [24], [25], [26]]
    
    dict_lst = {0: [0,1,2], 1: [3,4,5], 2: [6,7,8], 3: [9,10,11], 4: [12,13,14], 5: [15,16,17,18,19,20,21], 6: [22], 7: [23], 8: [24], 9: [25], 10: [26]}

    acquired = []
    for k in range(len(lst)):
        lst_to_be_checked = lst[k]
        if lst_to_be_checked[0] <= 21 and state[lst_to_be_checked[0]] != 1:
            acquired.append(lst_to_be_checked[0])
        elif lst_to_be_checked[0] > 21 and state[lst_to_be_checked[0]] != -1:
            acquired.append(lst_to_be_checked[0])

    dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

    num_features = list(set(dct_indx))

    return len(num_features)

def make_train(action,vec,i):
    lst = [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14], [15,16,17,18,19,20,21], [22], [23], [24], [25], [26]]
    
    dict_lst = {0: [0,1,2], 1: [3,4,5], 2: [6,7,8], 3: [9,10,11], 4: [12,13,14], 5: [15,16,17,18,19,20,21], 6: [22], 7: [23], 8: [24], 9: [25], 10: [26]}

    val = tuple([X_train_z.iloc[i,dict_lst[action][j]] for j,value in enumerate(dict_lst[action])])

    tup = tuple(vec[0])

    reward_state = reward(tup)
    
    tup = tup[:dict_lst[action][0]] + val + tup[dict_lst[action][-1]+1:]

    acquired = []
    for k in range(len(lst)):
        lst_to_be_checked = lst[k]
        if lst_to_be_checked[0] <= 21 and tup[lst_to_be_checked[0]] != 1:
            acquired.append(lst_to_be_checked[0])
        elif lst_to_be_checked[0] > 21 and tup[lst_to_be_checked[0]] != -1:
            acquired.append(lst_to_be_checked[0])

    dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

    num_features = list(set(dct_indx))

    is_terminal = len(num_features) == 11

    #mask = get_mask(np.asarray(tup))
    #reward_state = reward(tup)#,mask)

    return np.asarray(tup), reward_state, is_terminal #new node


def make_test(action,vec,i):
    lst = [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14], [15,16,17,18,19,20,21], [22], [23], [24], [25], [26]]
    
    dict_lst = {0: [0,1,2], 1: [3,4,5], 2: [6,7,8], 3: [9,10,11], 4: [12,13,14], 5: [15,16,17,18,19,20,21], 6: [22], 7: [23], 8: [24], 9: [25], 10: [26]}

    val = tuple([X_test_z.iloc[i,dict_lst[action][j]] for j,value in enumerate(dict_lst[action])])

    tup = tuple(vec[0])

    #reward_state = reward(tup)
    
    tup = tup[:dict_lst[action][0]] + val + tup[dict_lst[action][-1]+1:]

    acquired = []
    for k in range(len(lst)):
        lst_to_be_checked = lst[k]
        if lst_to_be_checked[0] <= 21 and tup[lst_to_be_checked[0]] != 1:
            acquired.append(lst_to_be_checked[0])
        elif lst_to_be_checked[0] > 21 and tup[lst_to_be_checked[0]] != -1:
            acquired.append(lst_to_be_checked[0])

    dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

    num_features = list(set(dct_indx))

    is_terminal = len(num_features) == 11
    #mask = get_mask(np.asarray(tup))

    #reward_state = reward(tup)#,mask)

    return np.asarray(tup), reward_state, is_terminal #new node

def cost(state):

    cost = 0
    #total_cost = 41

    lst = [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14], [15,16,17,18,19,20,21], [22], [23], [24], [25], [26]]
    
    dict_lst = {0: [0,1,2], 1: [3,4,5], 2: [6,7,8], 3: [9,10,11], 4: [12,13,14], 5: [15,16,17,18,19,20,21], 6: [22], 7: [23], 8: [24], 9: [25], 10: [26]}

    acquired = []
    for i in range(len(lst)):
        lst_to_be_checked = lst[i]
        if lst_to_be_checked[0] <= 21 and state[lst_to_be_checked[0]] != 1:
            acquired.append(lst_to_be_checked[0])
        elif lst_to_be_checked[0] > 21 and state[lst_to_be_checked[0]] != -1:
            acquired.append(lst_to_be_checked[0])

    dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

    num_features = list(set(dct_indx))

    for i in range(len(num_features)):
        if num_features[i] <= 5:
            cost += 1
        else:
            cost += 7

    return cost

# Parameters
# update_timestep - batch size
# update_target
# resetting epsilon - epsilon decay and resetting epsilon


# Standard practice is to update after each step. 
# DQN - L2 error between approximated and real (?) Q values. 
# Q values based on discounted rewards. 
# Approximation based on NN (given state). 
# Update based on randomly sampled states from memory. 
# 
# Why replay buffer, 


# logging variables
state_size = 27
action_size = 11
agent = DQNAgent(state_size, action_size)
done = False
batch_size = 6
num_epochs = 10
update_timestep = 6
update_target = 11
purge_step = 1000000#int(324/2.0)
retrain = 1000000

total_cost = 41.0
max_episodes = EPISODES     # max training episodes - this is okay
max_timesteps = action_size # max timesteps in one episode - this is okay

loss_total = []
reward_total = []

import heapq
import json

X_train_z.reset_index(inplace=True, drop=True)
X_train_z.columns = [i for i in range(0,27)]

X_test_z.reset_index(inplace=True, drop=True)
X_test_z.columns = [i for i in range(0,27)]

for j in range(num_epochs):
    
    print(j)

    reward_per_epoch = []
    loss_per_epoch =[]
    retrain_node = []
    X_train_z_copy = X_train_z.copy()
    y_train_copy = y_train.copy()
    purge_time = 1
    retrain_time = 1

    for i in range(m):
        
        print(i)

        loss_per_sample = []
        reward_per_sample = []

        timestep_replay = 0
        timestep_target = 0

        for e in range(EPISODES):

            act = []
            state = np.array([1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,-1,-1,-1,-1,-1])
            state = np.reshape(state, [1, state_size])

            for time_step in range(action_size):

                timestep_replay += 1
                timestep_target += 1
                
                action_values = agent.act(state)

                action = heapq.nlargest(action_size,range(len(action_values)), key=action_values.__getitem__)

                st =  set(act)

                to_be = [ele for ele in action if ele not in st]

                action = to_be[0]

                next_state, reward_state, done = make_train(action,state,i)

                act.append(action)

                print(act)

                #print(act)

                retrain_node.append(next_state)

                next_state = np.reshape(next_state, [1, state_size])
                agent.memorize(state, action, reward_state, next_state, done)
                state = next_state
                #mask = get_mask(state[0])
                reward_per_sample.append(reward_state)#,mask))

                if done:
                    break

                if timestep_target % update_target == 0:

                    agent.update_target_model()

                    timestep_target = 0

                if timestep_replay % update_timestep == 0:
                    #agent.update_target_model()

                    loss = agent.replay(batch_size)
                    loss_per_sample.append(loss)
                    #print('replaying')
                    timestep_replay = 0
                    #print(agent.epsilon)
                
            agent.reset_epsilon()

        if (i+1) % retrain == 0:
            
            print('Retraining')

            retrain_node_transform = []
            for r in range(len(retrain_node)):
                state_retrain = retrain_node[r]
                for u in range(len(state_retrain)):
                    if state_retrain[u] == -1.0:
                        state_retrain[u] = quad_end(total_cost,coeff,cost(state_retrain))#linear_function(coeff,total_cost,0,0,cost(state))#coeff#
                retrain_node_transform.append(state_retrain)

            indices = []
            for q in range(len(retrain_node_transform)):
                z = random.choice(range(len(retrain_node_transform))) #len from 0 to 157, after purge, indices should be from 157 to 324
                indices.append(z) #1 or 2 

            retrain_node_transform_selected = [retrain_node_transform[indices[z]] for z in range(len(indices))]

            retrain_node_transform_selected = pd.DataFrame(retrain_node_transform_selected)

            X_train_z_new = pd.concat([X_train_z_copy,retrain_node_transform_selected],ignore_index = True)

            y_train_new = []
            for k in range(len(retrain_node)):
                if purge_time > 1:
                    ind = int((indices[k]+purge_step*purge_time)/(max_episodes*max_timesteps)) #ind starts from 157 to 324 after purging
                else:
                    ind = int((indices[k])/(max_episodes*max_timesteps))
                y_train_new.append(y_train[ind])

            y_train_new = pd.DataFrame(y_train_new)

            y_train_new = pd.concat([pd.DataFrame(y_train_copy),y_train_new],ignore_index=True)

            # X_train_transform = []
            # for z in range(len(X_train_z_copy)):
            #     X = get_mask(np.array(X_train_z_copy.iloc[z,:]))
            #     X_train_transform.append(np.concatenate((np.array(X_train_z_copy.iloc[z,:]),X)))

            # X_test_total = []
            # for l in range(len(X_test_z)):
            #     X = get_mask(np.array(X_test_z.iloc[l,:]))
            #     X_test_total.append(np.concatenate((np.array(X_test_z.iloc[l,:]),X)))

            # X_train_transform = np.array(X_train_transform)
            # X_test_total = np.array(X_test_total)

            #y_train_copy= pd.DataFrame(y_train_copy)

            model_sk.fit(X_train_z_new, y_train_new)

            retrain_time += 1

            #retrain_node = []
        if (i + 1) % purge_step == 0:

            print('Purging')

            q,r = X_train_z_new.shape
            m,n = X_train_z.shape
            purge = q - m

            indx = random.sample(range(0, q), purge)

            indx_name = []

            for v in range(len(indx)):
                indx_name.append(X_train_z_new.iloc[indx[v],:].name)

            X_train_drop = X_train_z_new.drop(index=indx_name)

            y_train_drop = y_train_new.drop(index=indx_name)

            retrain_node = []

            X_train_z_copy = X_train_drop

            y_train_copy = y_train_drop.to_numpy()

            purge_time += 1
                
        loss_per_epoch.append(np.sum(loss_per_sample))
        reward_per_epoch.append(np.sum(reward_per_sample))
        agent.reset_epsilon()
    loss_total.append(np.mean(loss_per_epoch))
    reward_total.append(np.mean(reward_per_epoch))
    print(loss_total)
    print(reward_total)
    
    # name = 'loss_total_quadratic_end_pretrain_lr_dqn_hf_' + str(coeff) + '_' + str(j) + '_' + str(EPISODES)+ '.txt'
    # with open(name, 'w') as f:
    #     f.write(json.dumps(str(loss_total))) 

    # name = 'reward_total_quadratic_end_pretrain_lr_dqn_hf_' + str(coeff) + '_' + str(j) + '_' + str(EPISODES)+ '.txt'
    # with open(name, 'w') as f:
    #     f.write(json.dumps(str(reward_total)))

    name = 'DQN_model_weights_retrain_lr_hf_' + str(coeff) + '_' + str(j) + '_' + str(random_state) +  '_' + str(seed_value) + '.h5'
    agent.save(name)

    ep = agent.epsilon
    print(ep)
    agent.reset_epsilon()

#filename = 'retrained_quadratic_end_DQN_model_hf_smote_' + str(coeff) + '_' + str(random_state) + '_' + str(retrain) + '_1.sav'
#pickle.dump(model_sk, open(filename, 'wb'))

# m,n = X_train.shape
# state_total = []
# act_total = []
# for i in range(m):
#     print(i)
#     state = np.array([1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,-1,-1,-1,-1,-1])
#     state = np.reshape(state,[1,state_size])
#     act = []
#     done = False
#     while True:

#         if done:
#             break

#         action_values = agent.act(state)

#         action = heapq.nlargest(action_size,range(len(action_values)), key=action_values.__getitem__)

#         st =  set(act)

#         to_be = [ele for ele in action if ele not in st]

#         action = to_be[0]

#         next_state, reward_state, done = make_train(action,state,i)

#         next_state = np.reshape(next_state, [1, state_size])
#         state = next_state
#         state_total.append(state)

#         #print(action)

#         act.append(action)

#     act_total.append(act)

# m,n = X_test.shape
# state_total_test = []
# for i in range(m):
#     print(i)
#     state = np.array([1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,-1,-1,-1,-1,-1])
#     state = np.reshape(state,[1,state_size])
#     act = []
#     done = False
#     while True:

#         if done:
#             break

#         action_values = agent.act(state)

#         action = heapq.nlargest(action_size,range(len(action_values)), key=action_values.__getitem__)

#         st =  set(act)

#         to_be = [ele for ele in action if ele not in st]

#         action = to_be[0]

#         next_state, reward_state, done = make_test(action,state,i)

#         next_state = np.reshape(next_state, [1, state_size])
#         state = next_state
#         state_total_test.append(state)

#         act.append(action)

# state_total_list = []
# for i in range(len(state_total)):
#     node = tuple(np.array(state_total[i][0],dtype=np.float64))
#     state_total_list.append(node)

# state_test_total_list = []
# for i in range(len(state_total_test)):
#     node = tuple(np.array(state_total_test[i][0],dtype=np.float64))
#     state_test_total_list.append(node)

# import json
# name = 'node_train_quadratic_end_pretrain_lr_dqn_hf_' + str(random_state) + '_' + str(coeff) + '_2.txt'
# with open(name, 'w') as f:
#     f.write(json.dumps(state_total_list))

# name = 'node_test_quadratic_end_pretrain_lr_dqn_hf_' + str(random_state) + '_' + str(coeff) + '_2.txt'
# with open(name, 'w') as f:
#     f.write(json.dumps(state_test_total_list))
