"This script takes command line arguments and outputs the MNIST dataset and the respective models"

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

import sys

random_state = int(sys.argv[0])
seed_value = int(sys.argv[1])
nn = str(sys.argv[2])

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


if nn:

	x_train = x_train.reshape((-1,784))
	x_test = x_test.reshape((-1,784))

	x_train = pd.DataFrame(x_train)
	x_test = pd.DataFrame(x_test)

else:

	x_train = x_train.reshape((-1,28,28,1))
	x_test = x_test.reshape((-1,28,28,1))

#set number of categories
num_category = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)
 
# Output to file
print('random_state:\n', random_state)
print('seed_value:\n', seed_value)
print('CNN:\n', 'yes' if not nn else 'no' )

if nn:
	x_train_name = 'x_train_neural_net_' + str(random_state) + '_' + str(seed_value) + '.txt'
	x_test_name = 'x_test_neural_net_' + str(random_state) + '_' + str(seed_value) + '.txt'
	y_train_name = 'y_train_neural_net_' + str(random_state) + '_' + str(seed_value) + '.txt'
	y_test_name = 'y_test_neural_net_' + str(random_state) + '_' + str(seed_value) + '.txt'
else:
	x_train_name = 'x_train_cnn_' + str(random_state) + '_' + str(seed_value) + '.txt'
	x_test_name = 'x_test_cnn_' + str(random_state) + '_' + str(seed_value) + '.txt'
	y_train_name = 'y_train_cnn_' + str(random_state) + '_' + str(seed_value) + '.txt'
	y_test_name = 'y_test_cnn_' + str(random_state) + '_' + str(seed_value) + '.txt'

x_train_file = open(x_train_name, 'w+')
x_test_file = open(x_test_name, 'w+')
y_train_file = open(y_train_name, 'w+')
y_test_file = open(y_test_name, 'w+')

content = str(x_train)
x_train_file.write(content)
x_train_file.close()

content = str(x_train)
x_train_file.write(content)
x_train_file.close()

content = str(x_train)
x_train_file.write(content)
x_train_file.close()

# # Displaying the contents of the text file
# file = open("file2.txt", "r")
# content = file.read()
 
# print("\nContent in file2.txt:\n", content)
# file.close()

