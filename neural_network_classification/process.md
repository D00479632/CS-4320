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

### FIRST TRY

model_file: model1.keras

For this first try I am not going to change anything
Activation: relu
Layers: 1
Density: 100