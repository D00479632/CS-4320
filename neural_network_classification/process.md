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
872us/step - R2Score: 0.7000 - loss: 0.0444 - val_R2Score: 0.6934 - val_loss: 0.0461 - learning_rate: 0.1000

I called predict on this model because I wanted to submit it to kaggle to see how good it was.

I submitted predictions1.csv since it's like the sample submission and I got 0.93823 which is already pretty good.

### SECOND TRY

model_file: model1.keras
plot_file: learning-curve2.png

For this one now I am going to change some things and have:
Model: Sequential
Activation: relu for hidden layers and sigmoid for output
Layers: 2
Density: 128

It stopped and it was here:
epoc 14/100 991us/step - R2Score: 0.7229 - loss: 0.0408 - val_R2Score: 0.7044 - val_loss: 0.0444 - learning_rate: 0.0980
We have gotten better at R2 and also at val_R2 so I feel good about this model. 

Now its time that I do predict and see how good I did
The score was now 0.93980 which is a little better but not really

### THIRD TRY
