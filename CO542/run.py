from mnist import MNIST
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from main import * # importing created module
import numpy as np
import math as m


# importing datasets to data_dir and extract them.
datadir = './data'
mndata = MNIST(datadir)

# loading dataset
imgs, lbls = mndata.load_training()

examples = len(imgs)
pixel_size = m.sqrt(len(imgs[0]))
#print(examples, pixel_size)

# converting lists into arrays
images = np.array(imgs) # size - examples X pixel_size
labels = np.array(lbls) # size - examples X pixel_size

#print(type(labels))

# spliting dataset into train, test, validation
X_train, images2, Y_train, labels2 = train_test_split(images, labels, test_size=0.4, random_state=0)
X_validate, X_test, Y_validate, Y_test = train_test_split(images2, labels2, test_size=0.5, random_state=0)
# printing the precentages for train, test, validation

# print(len(X_train)*100/len(images))
# print(len(X_test)*100/len(images))
# print(len(X_validate)*100/len(images))

#print(mndata.display(images[0]))

# Preparing L_layer_model arguments

layer_dims = [784, 60, 10] # creating Layer Dimension list
learning_rate = 0.01 # assigning learning rate
itr = 30 # Iterations

parameters = L_layer_model(X_train.T, Y_train.T.reshape(1,36000), layer_dims, learning_rate, itr)
print(parameters)