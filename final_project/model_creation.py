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

def create_model_a(my_args, input_shape):
    """
    Create a model for diabetes prediction using tabular data.
    This model uses dense layers instead of convolutional layers
    since we're working with tabular data, not images.
    """
    model = keras.models.Sequential()
    
    model.add(keras.layers.Input(shape=input_shape))
    
    # First hidden layer
    model.add(keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal"))
    
    # Second hidden layer
    model.add(keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal"))
    
    model.add(keras.layers.Dropout(0.1))
    
    # Output layer (2 classes: diabetic, non-diabetic)
    model.add(keras.layers.Dense(2, activation="softmax"))
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["precision"]
    )
    
    return model
    
def create_model_b(my_args, input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))

    # First hidden layer
    model.add(keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    # Second hidden layer
    model.add(keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    # Third hidden layer
    model.add(keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.BatchNormalization())

    # Output layer
    model.add(keras.layers.Dense(2, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["Recall"]
    )

    return model

def create_model_c(my_args, input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))

    # First hidden layer
    model.add(keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    # Second hidden layer
    model.add(keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    # Third hidden layer
    model.add(keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.BatchNormalization())

    # Output layer
    model.add(keras.layers.Dense(2, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        # Lets see if the weights implemented in fit are enough to bump down false positives if not use:
        metrics=["Recall", "Precision"]
    )

    return model