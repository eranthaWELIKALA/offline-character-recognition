from input_data import *

file_dir = "D:\\Academic\\3rd Year\\6th SEM\\Projects\\Neural Networks\\Project-Source\\CO542"
mnist = read_data_sets(file_dir, one_hot=True)
train_images = mnist.train._images
print(type(train_images))
train_labels = mnist.train._labels
print(type(train_labels))
