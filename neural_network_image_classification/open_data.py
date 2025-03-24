#!/usr/bin/env python3

import numpy as np
import keras

#
# https://keras.io/api/datasets/fashion_mnist/
#
# 60,000 28x28 grayscale training images
# 10,000 28x28 grayscale testing images
#
# I choose to split them into 6 training batches,
# only using 1-5 to train and 6 to validate.
#

def load_fashion_mnist_batch_from_keras(number):
    """
    number in [1, 2, 3, 4, 5] -> load training batch
    number == 6 -> load validation batch
    number < 1 -> test batch
    number > 6 -> load training batches 1,2,3,4,5
    """
    # This is only the start, we will move to the CIFAR-10 dataset

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # shuffle training data (always the same seed)
    # Just in case the data comes organized
    np.random.seed(42)
    p = np.random.permutation(x_train.shape[0])
    x_train = x_train[p]
    y_train = y_train[p]

    # select subset
    if number < 1:
        images, labels = x_test, y_test
    elif number > 6:
        start = 0 * 10000
        end = 5 * 10000
        images, labels = x_train[start:end], y_train[start:end]
    else:
        start = (number - 1) * 10000
        end = (number) * 10000
        images, labels = x_train[start:end], y_train[start:end]

    # one-hot-encode labels
    labels = keras.utils.to_categorical(labels, num_classes=10)
    # reshape the (28,28) images to (28,28,1). Conv2D expects (w,h,depth)
    images = images.reshape(-1, 28, 28, 1)
    # scale images to [0,1] range
    images = images.astype(np.float32) / 255.0

    return images, labels

#
# https://keras.io/api/datasets/cifar10/
# 
# 50,000 32x32 color training images
# 10,000 32x32 color test images
# 10 categories.
#
# Now I will do a split into 5 training batches,
# using 1-4 to train and 5 to validate. Leaving the last portion
# of the data intact for finilized model.
#

def load_cifar10_batch_from_keras(number):
    """
    number in [1, 2, 3, 4] -> load training batch
    number == 5 -> load validation batch
    number < 1 -> test batch
    number > 5 -> load training batches 1,2,3,4
    """
    # The keras.load_data() returns two tuples with training and testing data

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # shuffle training data (always the same seed)
    # Just in case the data comes organized
    np.random.seed(42)
    p = np.random.permutation(x_train.shape[0])
    x_train = x_train[p]
    y_train = y_train[p]

    # select subset
    if number < 1:
        images, labels = x_test, y_test
    elif number > 5:
        start = 0 * 10000
        end = 4 * 10000
        images, labels = x_train[start:end], y_train[start:end]
    else:
        start = (number - 1) * 10000
        end = (number) * 10000
        images, labels = x_train[start:end], y_train[start:end]

    # one-hot-encode labels
    labels = keras.utils.to_categorical(labels, num_classes=10)
    # reshape the (32,32) images to (32,32,3). Conv2D expects (w,h,depth)
    images = images.reshape(-1, 32, 32, 3)
    # scale images to [0,1] range
    images = images.astype(np.float32) / 255.0

    return images, labels

def load_batch(number):
    return load_cifar10_batch_from_keras(number)
    #return load_fashion_mnist_batch_from_keras(number)

