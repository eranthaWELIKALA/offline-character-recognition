from mnist import MNIST
#from input_data import * # importing created module
import random

# importing datasets to data_dir and extract them.
datadir = './data'
mndata = MNIST(datadir)

# loading dataset
images, labels = mndata.load_training()
index = random.randrange(0, len(images))
print(type(images))
print(mndata.display(images[0]))