'''
This script implements the Proximal Policy Optimization algorithm on Pytorch
for the feature acquisition problem of MNIST. 

a) Each 4x4 block of pixels is defined as a feature with cost of 16
b) Feature acquisition episode starts from the empty feature state and ends in the full feature state
c) For PPO_AC, the policy update occurs at each timestep in an episode (49 timesteps)
d) For PPO_PG, the policy update occurs at end of an episode 

input: [1] = random state
input: [2] = seed value
input: [3] = Number of epochs for PPO training
input: [4] = update timestep (1 for AC and 49 for PG)
input: [5] = lr or cnn for classifier
input: [6] = Boolean for random strategy (optional)

outputs:

1) text files of loss and cumulative rewards at end of training
2) PPO policy weights at end of each epoch
3) Inference samples

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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

import itertools
import time
import json
import heapq

import sys

random_state = int(sys.argv[1])
seed_value = int(sys.argv[2])
num_epochs = int(sys.argv[3])
update_timestep = int(sys.argv[4])
classifier_name = sys.argv[5]

try:
    random_strategy = sys.argv[6].lower() == 'true'
    model_name = 'random'
except IndexError:
    random_strategy = False
    model_name = ''

if update_timestep == 1:
    ppo_name = 'ac'
else:
    ppo_name = 'pg'

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

# Set GPU support and random seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
t1 = torch.get_rng_state()
print(t1)

torch.set_rng_state(t1)

'''
class Memory: memory buffer for policy network update 

class ActorCritic: define the actor and critic policy networks

'''
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
        return action_probs 

    #for terminal states, return both action probabilities and their categorical distributions
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

        # Generalized advantage function
        def compute_gae(rewards, masks, values, next_values, gamma=0.99, tau=0.95):
            gae = 0
            returns = []
            for step in reversed(range(len(rewards))):
                delta = rewards[step] + gamma * next_values[step] * masks[step] - values[step] 
                # delta = r_T - V_T + r_T-1 - V_T-1 + ... + r_1 - V_1
                gae = delta + gamma * tau * masks[step] * gae
                returns.insert(0, gae)
            return returns

        # Shift an array to the right by one element
        def shift(arr, fill_value=0):
            result = np.empty_like(arr)

            result[:-1] = arr[1:]
            result[-1] = fill_value

            return result

        try:  
            # Normalizing the rewards:
            rewards = torch.tensor(memory.rewards[t:], dtype=torch.float32).to(device).detach()
        
            if len(rewards) != 1:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5) #this makes it from 0 to 1        
        
            # convert list to tensor
            old_states = torch.stack(memory.states[t:]).to(device).detach()
            old_actions = torch.stack(memory.actions[t:]).to(device).detach()
            old_logprobs = torch.stack(memory.logprobs[t:]).to(device).detach()
            is_terminals = torch.tensor(memory.is_terminals[t:]).to(device).detach()

        except RuntimeError: 
            # Normalizing the rewards:
            rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(device).detach()
        
            if len(rewards) != 1:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5) #this makes it from 0 to 1        
        
            # convert list to tensor
            old_states = torch.stack(memory.states).to(device).detach()
            old_actions = torch.stack(memory.actions).to(device).detach()
            old_logprobs = torch.stack(memory.logprobs).to(device).detach()
            is_terminals = torch.tensor(memory.is_terminals).to(device).detach()

        # Optimize policy for K epochs:
        i = 1
        pg_loss_total = 0.0
        val_loss_total = 0.0

        for _ in range(self.K_epochs):

            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            done = torch.tensor(memory.is_terminals, dtype=torch.float32).to(device)

            if len(done) == 1:
                ratios = torch.tensor(1)
            else:
                ratios = torch.exp(logprobs - old_logprobs.detach()) #make the ratio to be 1 for terminal

            # Finding Surrogate Loss:
            next_values = shift(state_values.detach().numpy())
            advantages = compute_gae(rewards,is_terminals.detach().numpy(),state_values.detach().numpy(),next_values,gamma=0.99,tau=0.95)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
            if len(advantages) != 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5) #this makes it from 0 to 1           

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + self.vl_coeff * self.MseLoss(state_values, rewards) - self.entropy_coeff * dist_entropy
            pg_loss = -torch.min(surr1, surr2).mean()
            val_loss = self.MseLoss(state_values,rewards)
            
            pg_loss_total += pg_loss
            val_loss_total += val_loss

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            i += 1
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        pg_loss_total /= i
        val_loss_total /= i

        loss_total = pg_loss_total + val_loss_total

        return loss_total, val_loss_total

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

############## Hyperparameters ##############
max_episodes = 100          # max training episodes
max_timesteps = 49          # max timesteps in one episode
n_latent_var = 256          # number of variables in hidden layer 
lr = 0.00001                # learning rate for the ActorCritic network
betas = (0.9, 0.999)        # beta1 and beta2 for Adam
gamma = 0.99                # discount factor
K_epochs = 4                # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
entropy_coeff = 0.02        # regularization parameter for the entropy loss
vl_coeff = 1.0              # regularization parameter for the value loss 
total_cost = 784.0          # total cost of full features
#############################################

# logging variables
state_dim = 784
action_dim = 49

m,n,o,p = x_train.shape

state_total = []
ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, entropy_coeff, vl_coeff)
memory = Memory()

val = []
loss = []
reward_total = []
history_total = []

#Training loop
for j in range(num_epochs):
    
    val_epoch = []
    loss_epoch = []
    reward_epoch = []


    # Define copies for retraining CNN
    
    # X_train_z_copy = x_train.copy()
    # y_train_copy = y_train.copy()

    # Define variables for retraining CNN
    
    # purge_time = 1
    # retrain_time = 1
    # retrain_node = []


    
    for i in range(1):#m):
        
        print(i)

        loss_per_sample = []
        val_per_sample = []            
        reward_per_sample = []

        #train loop
        for i_episode in range(max_episodes):

            timestep = 0

            act = []
            reward_per_episode = []
            loss_per_episode = []
            val_per_episode = []

            tp = (0.0,)*784
            state = np.array(tp,dtype=np.float64)

            done = False

            for t in range(max_timesteps):

                print(t)
                
                if done:
                    reward_per_sample.append(reward(state,total_cost,model_class,act)) #reward should be based on the current state
                    torch_state = torch.from_numpy(state).float()#.to(device) 
                    memory.rewards.append(reward(state,total_cost,model_class,act))
                    memory.is_terminals.append(done)
                    memory.states.append(torch_state)
                    action_probs, dist = ppo.policy_old.act_terminal(state, memory)
                    action = np.argmax(action_probs)

                    memory.actions.append(torch.tensor(action))
                    memory.logprobs.append(dist.log_prob(torch.tensor(action)))
                    break
                
                action_probs = ppo.policy_old.act(state, memory) #this always give 0,1,...,49

                dist = Categorical(action_probs)
                action_probs = action_probs.detach().numpy()

                action = heapq.nlargest(action_dim,range(len(action_probs)), key=action_probs.__getitem__)

                st =  set(act)

                to_be = [ele for ele in action if ele not in st]

                action = to_be[0]

                next_state, done = make(action,state,i,act,x_train) #make the next state based on selected action

                #retrain_node.append(state)

                # Save a list of action sequences up to time t
                act.append(action)

                # Saving reward and is_terminal:
                torch_state = torch.from_numpy(state).float()#.to(device)
                memory.states.append(torch_state)
                memory.rewards.append(reward(state,total_cost,model_class,act))
                memory.is_terminals.append(done)
                action = torch.tensor(action)
                memory.actions.append(action)
                memory.logprobs.append(dist.log_prob(action))

                reward_per_episode.append(reward(state,total_cost,model_class,act))

                state = next_state

            # update if its time
            for k in range(max_timesteps):

                timestep += 1

                # update if its time
                if timestep % update_timestep == 0:

                    loss_total, val_loss = ppo.update(memory,timestep)

                    val_per_episode.append(val_loss.detach().numpy())
                    loss_per_episode.append(loss_total.detach().numpy())

            #clear replay memory after one update cycle
            memory.clear_memory()
            reward_per_sample.append(np.sum(reward_per_episode))
            loss_per_sample.append(np.sum(loss_per_episode))
            val_per_sample.append(np.sum(val_per_episode))

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

        loss_epoch.append(np.mean(loss_per_sample))
        val_epoch.append(np.mean(val_per_sample))
        reward_epoch.append(np.mean(reward_per_sample))
    val.append(np.mean(val_epoch))
    loss.append(np.mean(loss_epoch))
    reward_total.append(np.mean(reward_epoch))
    print(val)
    print(loss)
    print(reward_total)

    # Save loss, reward, model weights
    if (j+1) % num_epochs == 0:

        val_name = 'loss_ppo_' + ppo_name + '_' + classifier_name + '_mnist_' + str(random_state) + '_' + str(j) + '_' + str(retrain) + '.txt'
        with open(val_name, 'w') as f:
            f.write(json.dumps(str(val)))

        reward_name = 'reward_ppo_' + ppo_name + '_' + classifier_name + '_mnist_' + str(random_state) + '_' + str(j) + '_' + str(retrain) + '.txt'
        with open(reward_name, 'w') as f:
            f.write(json.dumps(str(reward_total)))

    name = 'ppo_' + ppo_name + '_' + model_name + '_action_layer_' + classifier_name + '_mnist_' + str(random_state) + '_' + str(seed_value) + '_' + str(j) + '_' + str(retrain) + '.h5'
    torch.save(ppo.policy.action_layer.state_dict(),name)
    name = 'ppo_' + ppo_name + '_' + model_name + '_value_layer_' + classifier_name + '_mnist_' + str(random_state) + '_' + str(seed_value) + '_' + str(j) + '_' + str(retrain) + '.h5'
    torch.save(ppo.policy.value_layer.state_dict(),name)

# Inference step

m,n,o,p = x_test.shape
state_total = []
action_test_total = []
for i in range(1):#m):
    print(i)
    tp = (0.0,)*784
    state = np.array(tp,dtype=np.float64)        
    act = []
    done = False
    while True:

        state = torch.from_numpy(state).float()

        action_probs = ppo.policy.action_layer(state)

        action_probs = action_probs.detach().numpy()

        action = heapq.nlargest(action_dim,range(len(action_probs)), key=action_probs.__getitem__)

        st =  set(act)

        to_be = [ele for ele in action if ele not in st]

        action = to_be[0]

        state = state.numpy()

        next_state, done = make(action,state,i,act,x_test) #make a child

        state = next_state

        act.append(action)

        state_total.append(next_state.copy())

        done = len(act) == 49

        if done:

            state_total.append(next_state.copy())

            break

    action_test_total.append(act)

state_name = 'node_test_ppo_' + ppo_name + '_' + classifier_name + '_mnist_' + str(random_state) + '_' + str(seed_value) + '_' + str(num_epochs) + '_' + str(retrain) + '.txt'
with open(state_name, 'w') as f:
    f.write(json.dumps(str(state_total)))

action_name = 'feature_acquisition_action_sequence_test_ppo_' + ppo_name + '_' + classifier_name + '_mnist_' + str(random_state) + '_' + str(seed_value) + '_' + str(num_epochs) + '_' + str(retrain) + '.txt'
with open(action_name, 'w') as f:
    f.write(json.dumps(str(action_test_total)))