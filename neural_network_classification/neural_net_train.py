#!/usr/bin/env python3

import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

# TODO: change harcoded names
label = "loan_status"
input_filename = "loan-preprocessed-train.csv"
model_filename = "loan-model.keras"
train_ratio = 0.80
learning_curve_filename = "loan-learning-curve.png"
#
# Load the training dataframe, separate into X/y
#
dataframe = pd.read_csv(input_filename, index_col=0)
X = dataframe.drop(label, axis=1)
y = dataframe[label]

# print(dataframe)
# print(X)
# print(y)

# Tensor is like the abstraction of an array of data
#
# Prepare a tensorflow dataset from the dataframe
#
# Creates a tensorflow dataset that is an abstraction that allows you to 
# load files or dataframe and it slices it so that you don't have to store it 
# all in memory
dataset = tf.data.Dataset.from_tensor_slices((X, y)) # Make them a touple so that there is difference of label and feature
# print(dataset)
# print(list(dataset.as_numpy_iterator()))
# print(dataset.element_spec)


#
# Find the shape of the inputs and outputs.
# Necessary for the model to have correctly sized input and output layers
#
#
# This is happening *before* batching, so
# the shape does not yet include the batch size
#
for features, labels in dataset.take(1):
    input_shape = features.shape
    output_shape = labels.shape # This should be 1
# print(input_shape)
# print(output_shape)


#
# Split the dataset into train and validation sets.
#
dataset_size       = dataset.cardinality().numpy()
train_size         = int(train_ratio * dataset_size)
validate_size      = dataset_size - train_size
train_dataset      = dataset.take(train_size) # This would be the first 80% of the data
validation_dataset = dataset.skip(train_size) # This would be the remaining 20% of the data

#
# Cause the datasets to shuffle, internally
#
# This doesn't actually shuffle the data, it just sets up a pipeline to shuffle the data
# when you draw from this data
train_dataset      = train_dataset.shuffle(buffer_size=train_size)
validation_dataset = validation_dataset.shuffle(buffer_size=validate_size)

#
# Cause the datasets to batch.
# Efficiency benefits.
# Training differences.
#
BATCH_SIZE = 32
# Prefetching is important when files need to be grabbed from disk
train_dataset      = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



#
# Build the model
#
tf.random.set_seed(42)
# Sequential model is the simplest
model = keras.Sequential()
# input_shape needs to match the shape of the data bc the data needs to fit the model
model.add(keras.layers.Input(shape=input_shape))
model.add(keras.layers.Dense(100, activation="relu"))
# Output layer only has one unit in it (one output)
# Sigmoid is the threshold function type that helps for the steps
model.add(keras.layers.Dense(1, activation="sigmoid"))
#print(model.summary())
#print(model.layers[1].get_weights())

#
# Compile the model
#
# binary_crossentropy is typical for binary classification
loss = "binary_crossentropy"
model.compile(loss=loss,
              # Optimizer is the algorithm to use to try and find a min of the loss
              optimizer=keras.optimizers.SGD(learning_rate=0.1),
              # This is to track the area under the curve while learning
              metrics=["AUC"])


#
# Update the learning rate dynamically
#
def scheduler(epoch, learning_rate):
    r = learning_rate
    if epoch >= 10:
        r = learning_rate * float(tf.exp(-0.005))
    return r
learning_rate_callback = keras.callbacks.LearningRateScheduler(scheduler)

#
# Stop training if validation loss does not improve
#
# Callback functions are called each time an epoch is run 
# In this case if after 5 epochs the val_loss hasn't changed then we stop the epochs. Then because
# restore is True it will go 5 steps back and get that val_loss
early_stop_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

#
# Train for up to epoch_count epochs
#
epoch_count = 100
history = model.fit(x=train_dataset,
                    epochs=epoch_count,
                    validation_data=validation_dataset,
                    callbacks=[learning_rate_callback, early_stop_callback])
epochs = len(history.epoch)
# print(history)


#
# Display the learning curves
#
# Takes the history from fitting the model and plots the learning curve
line_style = ["r--", "r--*", "r--+", "b-", "b-*", "b-+"]
line_style = ["r--*", "r--+", "b-*", "b-+"]
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, epochs-1], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=line_style)
# plt.show()
plt.savefig(learning_curve_filename)
plt.clf()


#
# Save the model
#
model.save(model_filename)
