import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.linear_model import LogisticRegression as sk
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.regularizers import l1_l2

from tensorflow.keras.datasets import mnist

import pickle
import random

import itertools
import time

import json

random_state = 1

seed_value = 12321

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

x_train = x_train.reshape((-1,784))
x_test = x_test.reshape((-1,784))

#x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
#y_train = pd.DataFrame(y_train)
#y_test = pd.DataFrame(y_test)

#X_train_z = np.asarray(X_train_z)
y_train_new = []
X_train_new = []
for i in range(len(x_train)):
    z = x_train[i]
    random_num = np.random.choice(np.arange(0,2), p=[1/601, 600/601])
    if random_num == 0:
        X_train_new.append(z)
        y_train_new.append(y_train[i])

X_train_new = pd.DataFrame(X_train_new)
y_train_new = pd.DataFrame(y_train_new)

x_train = pd.DataFrame(x_train)

stride = [4,4]

#coeff = -90

filename = 'finalized_model_mnist_' + str(random_state) + '_' + str(seed_value) + '.sav'
#filename = 'finalized_model_chd_smote_random_' + str(random_state) + '_' + str(coeff) + '_' + str(seed_value) + '_quadratic_end.sav' #1_-90_12321_quadratic_end
#filename_fit = 'finalized_model_chd_fit_' + str(random_state) + '_' + str(seed_value) + '.sav'#1_123'

model_sk = pickle.load(open(filename, 'rb'))
#model_fit = pickle.load(open(filename_fit, 'rb'))

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

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.regularizers import l1_l2
# import pickle 

# tf.random.set_seed(seed_value)

# #import tensorflow as tf
# #from tensorflow import keras
# #filename = 'hf_random_' + str(random_state) + '_' + str(random_state) + '_.h5'
# #model_sk = keras.models.load_model(filename)#, custom_objects={'Functional':keras.models.Model})
# filename = 'chd' + '_' + str(random_state) + '_' + str(seed_value)# + '_quadratic_end'
# weight_name = filename + '_.h5'
# opt_name = filename + '_optimizer.pkl'

# initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal')
# inputs = keras.Input(shape=(40,), name="digits")
# x1 = layers.Dense(256, activation="relu",kernel_initializer=initializer)(inputs)
# x2 = layers.Dense(512, activation="relu",kernel_initializer=initializer)(x1)
# x3 = layers.Dense(128, activation="relu",kernel_initializer=initializer)(x2)
# outputs = layers.Dense(2,activation ='softmax',name="predictions",dtype='float64')(x3)
# model_sk = keras.Model(inputs=inputs, outputs=outputs)

# # # # Instantiate an optimizer.
# lr = 1e-6
# epochs = 101

# with open(opt_name, 'rb') as f:
#     weight_values = pickle.load(f)

# grad_vars = model_sk.trainable_weights

# optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

# zero_grads = [tf.zeros_like(w) for w in grad_vars]

# # Apply gradients which don't do nothing with Adam
# optimizer.apply_gradients(zip(zero_grads, grad_vars))

# # Set the weights of the optimizer
# optimizer.set_weights(weight_values)

# # # # NOW set the trainable weights of the model
# model_sk.load_weights(weight_name)

# # Instantiate a loss function.
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# # Prepare the metrics.
# train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
# val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# @tf.function
# def train_step(x, y):
#     with tf.GradientTape() as tape:
#         logits = model_sk(x, training=True)
#         loss_value = loss_fn(y, logits)
#     grads = tape.gradient(loss_value, model_sk.trainable_weights)
#     optimizer.apply_gradients(zip(grads, model_sk.trainable_weights))
#     train_acc_metric.update_state(y, logits)
#     return loss_value

# @tf.function
# def test_step(x, y):
#     val_logits = model_sk(x, training=False)
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

#         # if not any(i > val_acc for i in acc_test):
#         #     print('Saving')
#         #     model_name = "hf_" + str(random_state) + '_'  + str(random_state) + '_' + str(epoch)
#         #     symbolic_weights = getattr(model.optimizer, 'weights')
#         #     weight_values = K.batch_get_value(symbolic_weights)
#         #     opt_name = model_name + '_optimizer.pkl'
#         #     with open(opt_name, 'wb') as f:
#         #         pickle.dump(weight_values, f)
#         #     weight_name = model_name + '_.h5'
#         #     model.save_weights(weight_name)

#         acc_test.append(val_acc)

#         print("Validation acc: %.4f" % (float(val_acc),))
#         print("Validation loss: %.4f" % (float(loss_test_epoch),))

#         val_acc_metric.reset_states()

#         print("Time taken: %.2fs" % (time.time() - start_time))

#     return loss_train, loss_test, acc_train, acc_test

#import tensorflow as tf
#tf.random.set_seed(random_state)
#model_sk = tf.keras.models.load_model("hf_1_1.h5")

device = torch.device('cpu')#"cuda:0" if torch.cuda.is_available() else "cpu")
t1 = torch.get_rng_state()
print(t1)

# def processSubset(feature_set):
#     # Fit model on feature_set and calculate RSS
#     #model = sm.Logit(y_train,X_train.iloc[:,feature_set])
#     #model_sk = sk(max_iter=1000,class_weight = 'balanced',random_state = seed_value)
    
#     #regr = model_sk.fit(X_train_z.iloc[:,feature_set], y_train)

#     filename = 'hf_' + str(random_state) + '_' + str(seed_value) + '_' + str(feature_set) + '_.h5'

#     initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal')
#     inputs = keras.Input(shape=(len(feature_set),), name="digits")
#     x1 = layers.Dense(64,  kernel_regularizer=l1_l2(l1=1e-1, l2=1e-1), activation="relu",kernel_initializer=initializer)(inputs)
#     x2 = layers.Dense(32,  kernel_regularizer=l1_l2(l1=1e-1, l2=1e-1), activation="relu",kernel_initializer=initializer)(x1)
#     x3 = layers.Dense(16,  kernel_regularizer=l1_l2(l1=1e-1, l2=1e-1), activation="relu",kernel_initializer=initializer)(x2)
#     outputs = layers.Dense(2,activation ='softmax',name="predictions",dtype='float64')(x3)
#     model_fit = keras.Model(inputs=inputs, outputs=outputs)

#     model_fit.load_weights(filename)

#     return {tuple(feature_set): model_fit}

# def getBest(k):
    
#     tic = time.time()
    
#     results = []
            
#     #dict_lst = {0: [0,1,2], 1: [3,4,5,6,7,8], 2: [9,10,11,12,13], 3: [14,15,16], 4: [17,18,19,20,21], 5: [22,23,24], 6: [25,26,27], 7: [28,29,30], 8: [31,32,33], 9: [34], 10: [35], 11: [36], 12: [37], 13:[38], 14:[39]}

#     dict_lst = {0: [0,1,2], 1: [3,4,5], 2: [6,7,8], 3: [9,10,11], 4: [12,13,14], 5: [15,16,17,18,19,20,21], 6: [22], 7: [23], 8: [24], 9: [25], 10: [26]}

#     #dict_lst = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5,6,7,8,9,10,11], 6: [12,13,14], 7: [15,16,17], 8: [18,19,20], 9: [21,22,23], 10: [24,25,26]}

#     for combo in itertools.combinations([i for i in range(11)], k):
#         to_be_determined = []
#         for j in range(len(combo)):
#             to_be_determined.append(dict_lst[list(combo)[j]])
#         to_be_determined = [item for sublist in to_be_determined for item in sublist]
#         results.append(processSubset(to_be_determined))
    
#     result = dict((key,d[key]) for d in results for key in d)

#     # Wrap everything up in a nice dataframe
#     #models = pd.DataFrame(results)
    
#     # Choose the model with the highest RSS
#     #best_model = models.loc[models['RSS'].argmin()]
    
#     toc = time.time()
#     print("Processed", len(result), "models on", k, "predictors in", (toc-tic), "seconds.")
    
#     # Return the best model, along with some other useful information about the model
#     return result

# models_best = []

# tic = time.time()
# for i in range(1,12):
#     #getBest(i)
#     models_best.append(getBest(i))

# model_fit = dict((key,d[key]) for d in models_best for key in d)

# toc = time.time()
# print("Total elapsed time:", (toc-tic), "seconds.")

#torch.set_rng_state(torch.ByteTensor(12321))

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var*2),
                nn.Tanh(),
                nn.Linear(n_latent_var*2, int(n_latent_var/2)),
                nn.Tanh(),
                nn.Linear(int(n_latent_var/2), action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var*2),
                nn.Tanh(),
                nn.Linear(n_latent_var*2, int(n_latent_var/2)),
                nn.Tanh(),
                nn.Linear(int(n_latent_var/2), 1)
                )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        #dist = Categorical(action_probs)
        #action = dist.sample()
        
        #memory.states.append(state)
        #memory.logprobs.append(dist.log_prob(action))
        
        return action_probs #action.item(),dist#action_probs #action.item()

    def infer(self, state, memory):
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        #dist = Categorical(action_probs)
        #action = dist.sample()
        
        #memory.states.append(state)
        #memory.logprobs.append(dist.log_prob(action))
        
        return action_probs #action.item()

    def act_terminal(self, state):
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        return action_probs, dist

    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, state_value, dist_entropy
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, entropy_coeff, vl_coeff):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coeff = entropy_coeff
        self.vl_coeff = vl_coeff
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def update(self, memory,t):   
        # Monte Carlo estimate of state rewards:
        # rewards = []
        # discounted_reward = 0
        # for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
        #     if is_terminal:
        #         discounted_reward = 0
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)

        def compute_gae(rewards, masks, values, next_values, gamma=0.99, tau=0.95):
            gae = 0
            returns = []
            for step in reversed(range(len(rewards))):
                delta = rewards[step] + gamma * next_values[step] * masks[step] - values[step] # delta = r_T - V_T + r_T-1 - V_T-1 + ... + r_1 - V_1
                gae = delta + gamma * tau * masks[step] * gae
                returns.insert(0, gae)# + values[step])
            return returns

        def shift(arr, num, fill_value=0):
            result = np.empty_like(arr)

            result[:-1] = arr[1:]
            result[-1] = fill_value
            # if num > 0:
            #     result[:num] = fill_value
            #     result[num:] = arr[:-num]
            # elif num < 0:
            #     result[num:] = fill_value
            #     result[:num] = arr[-num:]
            # else:
            #     result[:] = arr
            return result

        # Normalizing the rewards:
        rewards = torch.tensor(memory.rewards[t:], dtype=torch.float32).to(device)
        if len(rewards) != 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5) #this makes it from 0 to 1        
        # convert list to tensor
        old_states = torch.stack(memory.states[t:]).to(device).detach()
        old_actions = torch.stack(memory.actions[t:]).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs[t:]).to(device).detach()

        #old_states = memory.states[t].to(device).detach()
        #old_actions = memory.actions[t].to(device).detach()
        #old_logprobs = memory.logprobs[t].to(device).detach()
        
        # Optimize policy for K epochs:
        i = 1
        pg_loss_total = 0.0
        val_loss_total = 0.0

        # logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

        # loss_fn = 1/2 * (old_states - (reward + gamma * state_values)) ** 2

        # self.optimizer_value_layer.zero_grad()
        # loss_fn.mean().backward()
        # self.optimizer_value_layer.step()


        for _ in range(self.K_epochs):

            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            #print(old_states)
            #print(old_actions)

            #print(logprobs)

            # Finding the ratio (pi_theta / pi_theta__old):


            # Finding the ratio (pi_theta / pi_theta__old):
            done = torch.tensor(memory.is_terminals, dtype=torch.float32).to(device)

            if len(done) == 1:
                ratios = torch.tensor(1)
            else:
                ratios = torch.exp(logprobs - old_logprobs.detach()) #make the ratio to be 1 for terminal

            # Finding Surrogate Loss:
            next_values = shift(state_values.detach().numpy(),-1,fill_value=0)
            advantages = compute_gae(rewards,memory.is_terminals[t:],state_values.detach().numpy(),next_values,gamma=0.99,tau=0.95)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
            if len(advantages) != 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5) #this makes it from 0 to 1            #advantages = rewards - state_values.detach()

            #print(ratios)
            #print(advantages)

            #advantages = gamma*state_values + reward - old_states  

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + self.vl_coeff * self.MseLoss(state_values, rewards) - self.entropy_coeff * dist_entropy
            pg_loss = -torch.min(surr1, surr2).mean()
            val_loss = self.MseLoss(state_values,rewards)
            
            pg_loss_total += pg_loss
            val_loss_total += val_loss

            #print(surr1)
            #print(surr2)
            #print(state_values)
            #print(rewards)
            #print(dist_entropy.mean())
            #print(pg_loss)

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            i += 1
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        pg_loss_total /= i
        val_loss_total /= i 

        return pg_loss_total, val_loss_total

def reward(state):

    state = np.asarray(state).flatten()

    cost = 0

    total_cost = 783.0

    lst = [[i] for i in range(0,784)]

    pairs = [[i,i] for i in range(0,784)]

    dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

    acquired = []
    for k in range(len(lst)):
        lst_to_be_checked = lst[k]
        if state[lst_to_be_checked[0]] != -1:
            acquired.append(lst_to_be_checked[0])

    dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

    num_features = list(set(dct_indx))

    for i in range(len(num_features)):
        cost += 1

    indx = [dict_lst[feature] for feature in dct_indx]
    flat_list = [item for sublist in indx for item in sublist]

    # if flat_list:
    #     if len(flat_list) < 40:
            
    #         state = np.array([val for i,val in enumerate(state) for j,index in enumerate(flat_list) if i == index])

    #         filename = 'chd_' + str(random_state) + '_' + str(seed_value) + '_' + str(flat_list) + '_.h5'
    #         initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal')
    #         inputs = keras.Input(shape=(len(flat_list),), name="digits")
    #         x1 = layers.Dense(256,  kernel_regularizer=l1_l2(l1=1e-1, l2=1e-1), activation="relu",kernel_initializer=initializer)(inputs)
    #         x2 = layers.Dense(128,  kernel_regularizer=l1_l2(l1=1e-1, l2=1e-1), activation="relu",kernel_initializer=initializer)(x1)
    #         x3 = layers.Dense(64,  kernel_regularizer=l1_l2(l1=1e-1, l2=1e-1), activation="relu",kernel_initializer=initializer)(x2)
    #         outputs = layers.Dense(2,activation ='softmax',name="predictions",dtype='float64')(x3)
    #         model = keras.Model(inputs=inputs, outputs=outputs)
    #         model.load_weights(filename)
    #     else:
    #         model = model_sk
    # else:
    model = model_sk

    for i in range(len(state)):
        if state[i] == -1:
            state[i] = quad_end(total_cost,coeff,cost)#linear_function(coeff,total_cost,0,0,cost)#coeff#linear_function(coeff,total_cost,0,0,cost)

    # X = state

    # W = model.get_weights()
    # X      = X.reshape((-1,X.shape[0]))           #Flatten
    # X      = X @ W[0] + W[1]                      #Dense
    # X[X<0] = 0                                    #Relu
    # X      = X @ W[2] + W[3]                      #Dense
    # X[X<0] = 0                                    #Relu
    # X      = X @ W[4] + W[5]                      #Dense
    # X[X<0] = 0                                    #Relu
    # X      = X @ W[6] + W[7]                      #Dense
    # X[X<0] = 0                                    #Relu
    # X      = np.exp(X)/np.exp(X).sum(1)[...,None] #Softmax
    # else:
    #     W = model.get_weights()
    #     X      = X.reshape((-1,X.shape[0]))           #Flatten
    #     X      = X @ W[0] + W[1]                      #Dense
    #     X[X<0] = 0                                    #Relu
    #     X      = X @ W[2] + W[3]                      #Dense
    #     X[X<0] = 0                                    #Relu
    #     X      = X @ W[4] + W[5]                      #Dense
    #     X[X<0] = 0                                    #Relu
    #     X      = np.exp(X)/np.exp(X).sum(1)[...,None] #Softmax
 
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

    return len(num_features)

def make_train(action,vec,i):
    
    lst = [[i] for i in range(0,784)]

    pairs = [[i,i] for i in range(0,784)]

    dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

    val = tuple([X_train_new.iloc[i,dict_lst[action][j]] for j, value in enumerate(dict_lst[action])])

    tup = tuple(vec)

    #reward_this_state = reward(tup)
    
    tup = tup[:dict_lst[action][0]] + val + tup[dict_lst[action][-1]+1:]

    acquired = []
    for k in range(len(lst)):
        lst_to_be_checked = lst[k]
        #if lst_to_be_checked[0] <= 33 and tup[lst_to_be_checked[0]] != 1:
        #    acquired.append(lst_to_be_checked[0])
        if tup[lst_to_be_checked[0]] != -1:
            acquired.append(lst_to_be_checked[0])

    dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

    num_features = list(set(dct_indx))

    is_terminal = len(num_features) == 784

    #reward_next_state = reward(tup) # reward of the next state

    return np.asarray(tup), is_terminal #new node


def make_test(action,vec,i):

    lst = [[i] for i in range(0,784)]

    pairs = [[i,i] for i in range(0,784)]

    dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

    val = tuple([x_test.iloc[i,dict_lst[action][j]] for j, value in enumerate(dict_lst[action])])

    tup = tuple(vec)

    #reward_this_state = reward(tup)
    
    tup = tup[:dict_lst[action][0]] + val + tup[dict_lst[action][-1]+1:]

    acquired = []
    for k in range(len(lst)):
        lst_to_be_checked = lst[k]
        #if lst_to_be_checked[0] <= 33 and tup[lst_to_be_checked[0]] != 1:
        #    acquired.append(lst_to_be_checked[0])
        if tup[lst_to_be_checked[0]] != -1:
            acquired.append(lst_to_be_checked[0])

    dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

    num_features = list(set(dct_indx))

    is_terminal = len(num_features) == 784

    #reward_next_state = reward(tup) # reward of the next state

    return np.asarray(tup), is_terminal #new node

def cost(state):

    cost = 0

    lst = [[i] for i in range(0,784)]

    pairs = [[i,i] for i in range(0,784)]

    dict_lst = dict([(k, [v]) for k, v in pairs])  #=> {'a': 2, 'b': 3}

    acquired = []
    for k in range(len(lst)):
        lst_to_be_checked = lst[k]
        if state[lst_to_be_checked[0]] != -1:
            acquired.append(lst_to_be_checked[0])

    dct_indx = [i for i, features in enumerate(lst) for j, val in enumerate(acquired) if val in features]

    num_features = list(set(dct_indx))

    for i in range(len(num_features)):
        cost += 1

    return cost

############## Hyperparameters ##############
max_episodes = 100          # max training episodes - this is okay
max_timesteps = 784         # max timesteps in one episode - this is okay
n_latent_var = 256          # number of variables in hidden layer - this needs to be optimized
update_timestep = 1         # update policy every n timesteps - this is okay
lr = 0.00001
betas = (0.9, 0.999)
gamma = 0.99                # discount factor
K_epochs = 4                # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
num_epochs = 1
#retrain = 27
entropy_coeff = 0.02
vl_coeff = 1.0
total_cost = 783.0
coeff = 0 #7,8,9,10
#############################################

# logging variables
state_dim = 784
action_dim = 784

m,n = X_train_new.shape

purge_step = 60000000#int(m/2.0)
retrain = 60000000

state_total = []
ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, entropy_coeff, vl_coeff)
memory = Memory()

val = []
loss = []
reward_total = []

history_total = []

indx = []
for j in range(10):
    l = np.where(y_train == j)
    l = l[0][0]
    indx.append(l)

import heapq
import time

for j in range(num_epochs):
    
    print(j)

    val_epoch = []
    loss_epoch = []
    reward_epoch = []
    retrain_node = []
    X_train_z_copy = X_train_new.copy()
    y_train_copy = y_train_new.copy()
    purge_time = 1
    retrain_time = 1
    
    for i in range(len(indx)):#m):
        
        print(i)

        timestep = 0
        loss_per_sample = []
        val_per_sample = []            
        reward_per_sample = []

        #train loop
        for i_episode in range(max_episodes):

            act = []
            #action = 12
            tp = (-1,)*784

            state = np.array(tp,dtype=np.float64)

            for t in range(max_timesteps):
                
                timestep += 1

                done = num_features(state) == action_dim

                if done:
                    reward_per_sample.append(reward(state)) #reward should be based on the current state
                    torch_state = torch.from_numpy(state).float().to(device) 
                    memory.rewards.append(reward(state))
                    memory.is_terminals.append(done)
                    memory.states.append(torch_state)
                    action_probs, dist = ppo.policy_old.act_terminal(state, memory)
                    action = np.argmax(action_probs)

                    memory.actions.append(torch.tensor(action))
                    memory.logprobs.append(dist.log_prob(torch.tensor(action)))
                    #memory.actions.append(torch.tensor(0))
                    #memory.logprobs.append(dist.log_prob(torch.tensor(0)))
                    break
                
                # Running policy_old:
                #action = 13
                #print(action)
                #while action in act:
                action_probs = ppo.policy_old.act(state, memory) #this always give 0,1,...,14
                    #print(action)
                #act.append(action)
                dist = Categorical(action_probs)
                action_probs = action_probs.detach().numpy()

                action = heapq.nlargest(action_dim,range(len(action_probs)), key=action_probs.__getitem__)

                st =  set(act)

                to_be = [ele for ele in action if ele not in st]

                action = to_be[0]

                #action_make = action.detach().numpy()

                next_state, done = make_train(action,state,indx[i]) #make a child

                #state = next_state

                retrain_node.append(state)

                act.append(action)

                #print(act)

                # Saving reward and is_terminal:
                torch_state = torch.from_numpy(state).float().to(device)
                memory.states.append(torch_state)
                memory.rewards.append(reward(state))
                memory.is_terminals.append(done)
                action = torch.tensor(action)
                memory.actions.append(action)
                memory.logprobs.append(dist.log_prob(action))

                reward_per_sample.append(reward(state))

                state = next_state

                # update if its time
            for k in range(max_timesteps):

                timestep += 1

                # update if its time
                if timestep % update_timestep == 0:

                    pg_loss, val_loss = ppo.update(memory,k)
                    #memory.clear_memory()
                    timestep = 0

                    #print(val_loss)
                    #print(pg_loss)

                    val_per_sample.append(val_loss.detach().numpy())
                    loss_per_sample.append(pg_loss.detach().numpy())

                # if timestep % update_timestep == 0:

                #     pg_loss, val_loss = ppo.update(memory,t)
                #     #memory.clear_memory()
                #     timestep = 0

                #     val_per_sample.append(val_loss.detach().numpy())
                #     loss_per_sample.append(pg_loss.detach().numpy())

                #if done:
                #    break
            memory.clear_memory()


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

            y_train_new_z = []
            for k in range(len(retrain_node)):
                if purge_time > 1:
                    ind = int((indices[k]+purge_step*purge_time)/(max_episodes*max_timesteps)) #ind starts from 157 to 324 after purging
                else:
                    ind = int((indices[k])/(max_episodes*max_timesteps))
                y_train_new_z.append(y_train_new.iloc[ind])

            y_train_new_z = pd.DataFrame(y_train_new_z)

            y_train_new_z = pd.concat([pd.DataFrame(y_train_copy),y_train_new_z],ignore_index=True)

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

            model_sk.fit(X_train_z_new, y_train_new_z)

            # batch_size = 512
            # train_X_total = np.reshape(X_train_z_new, (-1, 40))
            # test_X_total = np.reshape(X_test_z, (-1, 40))
            # train_dataset = tf.data.Dataset.from_tensor_slices((train_X_total, y_train_new_z))
            # train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

            # val_dataset = tf.data.Dataset.from_tensor_slices((X_test_z, y_test))
            # val_dataset = val_dataset.batch(batch_size)

            # loss_train, loss_test, acc_train, acc_test = train(train_dataset,val_dataset)

            #retrain_node = []

        if (i + 1) % purge_step == 0:

            print('Purging')

            q,r = X_train_z_new.shape
            m,n = X_train_new.shape
            purge = q - m

            indx = random.sample(range(0, q), purge)

            indx_name = []

            for t in range(len(indx)):
                indx_name.append(X_train_z_new.iloc[indx[t],:].name)

            X_train_drop = X_train_z_new.drop(index=indx_name)

            y_train_drop = y_train_new_z.drop(index=indx_name)

            retrain_node = []

            X_train_z_copy = X_train_drop

            y_train_copy = y_train_drop.to_numpy()

            purge_time += 1

        loss_epoch.append(np.sum(loss_per_sample))
        val_epoch.append(np.sum(val_per_sample))
        reward_epoch.append(np.sum(reward_per_sample))
    val.append(np.mean(val_epoch))
    loss.append(np.mean(loss_epoch))
    reward_total.append(np.mean(reward_epoch))
    print(val)
    print(loss)
    print(reward_total)
    # if (j+1) % num_epochs == 0:

    #     val_list = []
    #     for i in range(len(val)):
    #         val_node = tuple(np.array(val[i],dtype=np.float64))
    #         val_list.append(val_node)

    #     reward_list = []
    #     for i in range(len(reward_total)):
    #         reward_node = tuple(np.array(reward_total[i],dtype=np.float64))
    #         reward_list.append(reward_node)

    #     val_name = 'loss_ppo_ac_quadratic_pretrain_end_lr_physionet_' + str(random_state) + '_' + str(coeff) + '_' + str(j) + '_' + str(retrain) + '.txt'
    #     with open(val_name, 'w') as f:
    #         f.write(json.dumps(str(val)))

    #     reward_name = 'loss_ppo_ac_quadratic_pretrain_end_lr_physionet_' + str(random_state) + '_' + str(coeff) + '_' + str(j) + '_' + str(retrain) + '.txt'
    #     with open(reward_name, 'w') as f:
    #         f.write(json.dumps(str(reward)))

    name = 'ppo_ac_pretrain_quadratic_end_action_layer_lr_mnist_' + str(random_state) + '_' + str(seed_value) + '_' + str(coeff) + '_' + str(j) + '_' + str(retrain) + '_.h5'
    torch.save(ppo.policy_old.action_layer.state_dict(),name)
    name = 'ppo_ac_pretrain_quadratic_end_value_layer_lr_mnist_' + str(random_state) + '_' + str(seed_value) + '_' + str(coeff) + '_' + str(j) + '_' + str(retrain) + '_.h5'
    torch.save(ppo.policy_old.value_layer.state_dict(),name)
    #model.save_weights('model_' + str(j) + '.h5')

# filename = 'retrained_model_physionet_lr_balanced_ppo_ac_end_' + str(coeff) + '_' + str(random_state) + '_' + str(retrain) + '_' + str(seed_value) + '.sav'
# #model_sk.save_weights(filename)
# pickle.dump(model_sk, open(filename, 'wb'))

# X_train_z = pd.DataFrame(X_train_z)

# m,n = X_train_new.shape
# state_total = []
# action_train_total = []
# for i in range(m):
#     print(i)
#     state = np.array([1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
#     memory_test = Memory()
#     act = []
#     done = False
#     while True:

#         action_probs = ppo.policy.infer(state, memory_test) #this always give 0,1,...,14

#         action_probs = action_probs.detach().numpy()

#         action = heapq.nlargest(action_dim,range(len(action_probs)), key=action_probs.__getitem__)

#         st =  set(act)

#         to_be = [ele for ele in action if ele not in st]

#         action = to_be[0]

#         print(action)

#         next_state = make_train(action,state,i) #make a child

#         state_total.append(state)

#         state = next_state

#         num_node = num_features(state)

#         done = num_node == len(action_probs)

#         act.append(action)
        
#         if done:
#             break

#     memory_test.clear_memory()
#     #state_total.append(state)
#     action_train_total.append(act)

# m,n = X_test.shape
# state_test_total = []
# action_total = []
# for i in range(m):
#     print(i)
#     state_test = np.array([1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
#     memory_test = Memory()
#     act = []
#     done = False
#     while True:

#         action_probs = ppo.policy.infer(state_test, memory_test) #this always give 0,1,...,14

#         action_probs = action_probs.detach().numpy()

#         action = heapq.nlargest(action_dim,range(len(action_probs)), key=action_probs.__getitem__)

#         st =  set(act)

#         to_be = [ele for ele in action if ele not in st]

#         action = to_be[0]
#         #action = np.argmax(action_probs)

#         print(action)

#         next_state, done = make_test(action,state_test,i) #make a child

#         state_test_total.append(state_test)

#         state_test = next_state
        
#         act.append(action)

#         num_node = num_features(state_test)

#         done = num_node == len(action_probs)

#         if done:
#             break

#     memory_test.clear_memory()
#     #state_test_total.append(state_test)
#     action_total.append(act)

# state_total_list = []
# for i in range(len(state_total)):
#     node = tuple(np.array(state_total[i],dtype=np.float64))
#     state_total_list.append(node)

# state_test_total_list = []
# for i in range(len(state_test_total)):
#     node = tuple(np.array(state_test_total[i],dtype=np.float64))
#     state_test_total_list.append(node)

# import json
# name = 'node_train_pretrain_lr_balanced_ppo_ac_physionet_end_' + str(random_state) + '_' + str(coeff) + '_' + str(seed_value) + '_' + str(retrain) + '.txt'
# with open(name, 'w') as f:
#     f.write(json.dumps(state_total_list))

# name = 'node_test_pretrain_lr_balanced_ppo_ac_physionet_end_' + str(random_state) + '_' + str(coeff) + '_' + str(seed_value) + '_' + str(retrain) + '.txt'
# with open(name, 'w') as f:
#     f.write(json.dumps(state_test_total_list))

# filename = 'train_loss.txt' 
# with open(filename, 'w') as f:
#     for item in loss:
#         f.write("%s\n" % item)

# filename = 'train_val.txt' 
# with open(filename, 'w') as f:
#     for item in val:
#         f.write("%s\n" % item)

# filename = 'node_train.txt' 
# with open(filename, 'w') as f:
#     for item in state_total:
#         f.write("%s\n" % item)

# filename = 'node_test.txt' 
# with open(filename, 'w') as f:
#     for item in state_test_total:
#         f.write("%s\n" % item)