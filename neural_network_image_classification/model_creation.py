#!/usr/bin/env python3

#
# Keep the model creation code contained here
#
import tensorflow as tf
import keras

# Check if GPU is available
gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu_available else "NOT AVAILABLE")

def create_model(my_args, input_shape):
    """
    Control function.
    Selects the correct function to build a model, based on the model name
    from the command line arguments.

    Assumes my_args.model_name is set.
    """
    # Add the different models here
    create_functions = {
        "a": create_model_a,
        "b": create_model_b,
        "c": create_model_c,
        "d": create_model_d,
        "e": create_model_e,
        "f": create_model_f,
    }
    if my_args.model_name not in create_functions:
        raise Exception("Invalid model name: {} not in {}".format(my_args.model_name, list(create_functions.keys())))
        
    model = create_functions[my_args.model_name](my_args, input_shape)
    print(model.summary())
    return model


### Various model architectures (keep adding _b, _c, etc for the report)

'''
Suggested layer types are:

- [Conv2D](https://keras.io/api/layers/convolution_layers/convolution2d/)
- [MaxPooling2D](https://keras.io/api/layers/pooling_layers/max_pooling2d/)
- [Dense](https://keras.io/api/layers/core_layers/dense/)
- [Dropout](https://keras.io/api/layers/regularization_layers/dropout/)
- [Flatten](https://keras.io/api/layers/reshaping_layers/flatten/)
'''

'''
Suggested model structure is:

- Conv (bigger kernel size)
Bigger kernel size to make sure we capture large scale info before the first pooling.
- Pooling

-----------
- Conv (smaller kernel size)
- Conv (smaller kernel size)
Now we want to pay more attention to small collections of things so thats why smaller kernel size.
- Pooling
-----------
^ as many as wanted

-----------
- Flatten
- Dense
-----------
^ as many as wanted

- Output

(You can also add some dropout layers to prevent overfitting)
'''

'''
Select an optimizer and a learning rate. Consider the following
hyperparameters:

- Optimizer
- Batch size
- Number of epochs
- Learning rate
- Learning rate decay
'''

'''
Use the loss function that works best for you. Remember that this
guides the learning process. It is a hyperparameter that can
be tuned.
'''


# This model is for fashion_mnist dataset, don't use it after I change the files to work with CIFAR10 
def create_model_a(my_args, input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(7,7), activation="relu", kernel_initializer="he_normal", padding="same"))
    # In this case, MaxPooling2D will take a 2x2 region from the layer above and take the max of the 4 numbers to feed as output
    # So it reduces the width and height
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.Dropout(0.5))
    # Gives you probs that the image belongs to a specific class (the sum of all is 1)
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=keras.optimizers.Adam())
    return model

# My very first model with CIFAR10 will be just like the past one
def create_model_b(my_args, input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(7,7), activation="relu", kernel_initializer="he_normal", padding="same"))
    # In this case, MaxPooling2D will take a 2x2 region from the layer above and take the max of the 4 numbers to feed as output
    # So it reduces the width and height
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.Dropout(0.5))
    # Gives you probs that the image belongs to a specific class (the sum of all is 1)
    model.add(keras.layers.Dense(10, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    # Using categorical_crossentropy because we one-hot-encode the labels on open_data.py
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
    return model

def create_model_c(my_args, input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    
    # First convolutional layer with a larger kernel size
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(7,7), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    
    # Additional convolutional layers with smaller kernel sizes
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(5,5), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    
    # Additional convolutional layer for more feature extraction
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.Dropout(0.5))  # Dropout layer to prevent overfitting
    model.add(keras.layers.Dense(10, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
    return model

def create_model_d(my_args, input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    
    # First convolutional layer with a larger kernel size
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(7,7), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    
    # Additional convolutional layers with smaller kernel sizes
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(5,5), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.Dropout(0.5))  # Dropout layer to prevent overfitting
    model.add(keras.layers.Dense(10, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
    return model

def create_model_e(my_args, input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    
    # First convolutional layer with a larger kernel size
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(7,7), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    
    # Additional convolutional layers with smaller kernel sizes
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(5,5), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.25))  # Dropout layer to prevent overfitting
    
    # Additional convolutional layer for more feature extraction
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.35))  # Dropout layer to prevent overfitting
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.Dropout(0.15))  # Dropout layer to prevent overfitting
    model.add(keras.layers.Dense(10, activation="softmax"))

    # Read online that weight_decay=1e-4, helps with generalization
    optimizer = keras.optimizers.Adamax(learning_rate=0.001, weight_decay=1e-4)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
    return model

def create_model_f(my_args, input_shape):

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))

    model.add(keras.layers.Conv2D(64, (5,5), padding="same", kernel_initializer="he_normal"))
    # Trying this from last assignment
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(keras.layers.Conv2D(128, (5,5), padding="same", kernel_initializer="he_normal"))
    # Trying this from last assignment
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))

    model.add(keras.layers.Conv2D(128, (3,3), padding="same", kernel_initializer="he_normal"))
    # Trying this from last assignment
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    # Found this on keras and sounded interesting
    model.add(keras.layers.SpatialDropout2D(0.25))

    model.add(keras.layers.Conv2D(256, (3,3), padding="same", kernel_initializer="he_normal"))
    # Trying this from last assignment
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, kernel_initializer="he_normal"))
    # Trying this from last assignment
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Dense(10, activation="softmax"))

    optimizer = keras.optimizers.Adamax(learning_rate=0.001, weight_decay=1e-4)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)

    return model
