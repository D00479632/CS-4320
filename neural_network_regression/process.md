# Classification (Using Neural Networks)

[Download data here](https://www.kaggle.com/competitions/playground-series-s4e11/data?select=test.csv)


## Visualize the data

I have loaded all the starter code and looked at the data to split it into categorical and numberical. 

It looks like we are dealing with 18 features and to me all of them seem useful for predicting the outcome but 
the feature "name". I don't really see how that could affect if a person had depression or not so in future 
fine tunings I might take it out.

## PREPARE DATA

I will use the `preprocess.py` to get my preprocess data since sklearn has better preprocessing methos than keras
or tensorflow.

Once I have my new preprocess data I want to make sure that the sizes are the same: 

```bash
wc -l data/*.csv
   93801 data/preprocessed-test.csv
  140701 data/preprocessed-train.csv
   93801 data/sample_submission.csv
   93801 data/test.csv
  140701 data/train.csv
  562805 total
```

## SELECT/TRAIN MODEL

Now we will be working with the file `neural_net_train.py`
The loss will always be the same: "mean_squared_error"
The metrics will always be the same: "R2Score"
The optimizer will always be SGD

### FIRST TRY

model_file: model1.keras
plot_file: learning-curve1.png

For this first try I am not going to change anything
Model: sequential
Activation: relu
Layers: 1
Density: 100

This is the run that it stopped at and i think that the loss is pretty good so I am expecting a decent score.
Epoch 8/100 872us/step - R2Score: 0.7000 - loss: 0.0444 - val_R2Score: 0.6934 - val_loss: 0.0461 - learning_rate: 0.1000

I called predict on this model because I wanted to submit it to kaggle to see how good it was.

I submitted predictions1.csv since it's like the sample submission and I got 0.93823 which is already pretty good.

### SECOND TRY

model_file: model1.keras
plot_file: learning-curve2.png

For this one now I am going to change some things and have:
Model: Sequential
Activation: relu for hidden layers and sigmoid for output
Initializer = HeNormal()
Layers: 2
Density: 128

It stopped and it was here:
epoc 14/100 991us/step - R2Score: 0.7229 - loss: 0.0408 - val_R2Score: 0.7044 - val_loss: 0.0444 - learning_rate: 0.0980
We have gotten better at R2 and also at val_R2 so I feel good about this model. 

Now its time that I do predict and see how good I did
The score was now 0.93980 which is a little better but not really

### THIRD TRY

I am just going to try with the same exact last model but I am going to remove the label name out of the preprocessed data
Model: Sequential
Activation: relu for hidden layers and sigmoid for output
Initializer = HeNormal()
Layers: 2
Density: 128

Epoch 14/100 842us/step - R2Score: 0.7102 - loss: 0.0432 - val_R2Score: 0.7023 - val_loss: 0.0447 - learning_rate: 0.0980
There is barely any difference, I am going to try and make the batches bigger to 64 since we have one less column
Epoch 12/100 1ms/step - R2Score: 0.7054 - loss: 0.0440 - val_R2Score: 0.7043 - val_loss: 0.0444 - learning_rate: 0.0990
We have not done any better so I'll keep it the way it was
Epoch 8/100 892us/step - R2Score: 0.7067 - loss: 0.0437 - val_R2Score: 0.7031 - val_loss: 0.0446 - learning_rate: 0.1000
This time it took less epochs and the R2 is worse than the first time I trained it

```bash
./neural_net_predict.py
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 128)                 │          45,568 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 128)                 │          16,512 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 1)                   │             129 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 62,211 (243.02 KB)
 Trainable params: 62,209 (243.00 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 2 (12.00 B)
None
Traceback (most recent call last):
  File "/Users/paulalozanogonzalo/spring2025/CS-4320/neural_network_classification/./neural_net_predict.py", line 52, in <module>
    y_hat = model.predict(X)
            ^^^^^^^^^^^^^^^^
  File "/Users/paulalozanogonzalo/spring2025/CS-4320/.venv11/lib/python3.11/site-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/Users/paulalozanogonzalo/spring2025/CS-4320/.venv11/lib/python3.11/site-packages/keras/src/layers/input_spec.py", line 227, in assert_input_compatibility
    raise ValueError(
ValueError: Exception encountered when calling Sequential.call().

Input 0 of layer "dense" is incompatible with the layer: expected axis -1 of input shape to have value 355, but received input with shape (32, 777)

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=(32, 777), dtype=float32)
  • training=False
  • mask=None
```

For some reason I can't predict and I cannot understan why. I am just going to pass on this one and try something new

### FOURTH TRY

```python
model.add(keras.layers.Input(shape=input_shape))

activation = 'selu'
initializer = keras.initializers.LecunNormal()
layers = 3
for i in range(layers):
    model.add(keras.layers.Dense(100, activation=activation, kernel_initializer=initializer))
    model.add(keras.layers.BatchNormalization())  # Normalizes activations for stability
    model.add(keras.layers.Dropout(0.3))  # Prevents overfitting

model.add(keras.layers.Dense(1, activation="sigmoid"))
```

I added 9 layers and an attention layer. I read over the documentation and these were interesting so lets see if I improve insteas of overfitting.

Best epoch: Epoch 9/100 1ms/step - R2Score: 0.6649 - loss: 0.0496 - val_R2Score: 0.7042 - val_loss: 0.0444 - learning_rate: 0.1000
Last epoch: Epoch 14/100 1ms/step - R2Score: 0.6670 - loss: 0.0495 - val_R2Score: 0.7035 - val_loss: 0.0445 - learning_rate: 0.0980

I don't think this will be better but I will predict anyway. Looking at the learning curve its visible that it won't do as good.
Never mind, it got a 0.93937 so this is my second best model