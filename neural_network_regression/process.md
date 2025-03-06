# Classification (Using Neural Networks)

[Download data here](https://www.kaggle.com/competitions/playground-series-s4e12/overview)


## Visualize the data

I have loaded all the starter code from the past assignment and looked at the data to split it into categorical and numberical. 

It looks like we are dealing with 19 features and to me all of them seem useful for predicting the outcome but 
the feature "Policy Start Date". I don't really know how to deal with that feature since its very specific, for example:
2023-12-23 15:21:39.134960
We are dealing with years, months, days, hours, minutes and seconds so in order to make it either categorical or numerical I would have
to pre-preprocess this feature but since I don't think its very important I will just not add it into my pipeline. If it was something like
"Vehicle Age" I would for sure do something about it because I think that this feature does change the output by a lot.

This is the way I am splitting my features into categorical and numerical:
```python
self.mCategoricalPredictors = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location', 'Policy Type', 
                                        'Customer Feedback', 'Smoking Status', 'Exercise Frequency', 'Property Type']
        self.mNumericalPredictors = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'Previous Claims', 'Vehicle Age', 
                                        'Credit Score', 'Insurance Duration']
```

## PREPARE DATA

I will use the `preprocess.py` to get my preprocess data since sklearn has better preprocessing methos than keras
or tensorflow.

Once I have my new preprocess data I want to make sure that the sizes are the same to make sure we didn't loose any row: 

```bash
wc -l data/*.csv
  800001 data/preprocessed-test.csv
 1200001 data/preprocessed-train.csv
  800001 data/test.csv
 1200001 data/train.csv
```

## SELECT/TRAIN MODEL

Now we will be working with the file `neural_net_train.py`
To see what loss we should use the best way is to check what they use to evaluate it:
Submissions are evaluated using the Root Mean Squared Logarithmic Error (RMSLE)
Going onto keras and look for the [RMSLE](https://keras.io/api/losses/regression_losses/#meansquaredlogarithmicerror-class)
```python
class MeanSquaredLogarithmicError(LossFunctionWrapper):
    """Computes the mean squared logarithmic error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(square(log(y_true + 1) - log(y_pred + 1)))
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.
    """
```
The loss will always be the same: "keras.losses.MeanSquaredLogarithmicError(
    reduction="sum_over_batch_size", name="mean_squared_logarithmic_error", dtype=None
)
"
The metrics will always be the same: "keras.metrics.MeanSquaredLogarithmicError(
    name="mean_squared_logarithmic_error", dtype=None
)
keras.metrics.Accuracy()
"
The optimizer will always be SGD

### FIRST TRY

model_file: model1.keras
plot_file: learning-curve1.png

For this first try I am not going to change anything
Model: sequential
Activation: relu and linear
Layers: 1
Density: 100
Batch Size: 32

My model didn't converge, it ran for the 100 epochs which took 40 minutes. I am worried that I am overfitting the 
training data but I only have one layer so it shouldn't.

Epoch 100/100
30000/30000 ━━━━━━━━━━━━━━━━━━━━ 23s 728us/step - accuracy: 0.0000e+00 - loss: 1.1593 - mean_squared_logarithmic_error: 1.1593 - val_accuracy: 0.0000e+00 - val_loss: 1.1506 - val_mean_squared_logarithmic_error: 1.1506 - learning_rate: 0.0638

I called predict on this model because I wanted to submit it to kaggle to see how good it was.

I submitted predictions_proba1.csv since it's like the sample submission and I got 1.07819 which is already pretty good since the requirement is to get  below 1.12 on 
the hidden data set.

### SECOND TRY

model_file: model2.keras
plot_file: learning-curve2.png

For this one now I am going to change some things and have:
Model: Sequential
Activation: swish for hidden layers and linear for output
Initializer = he_normal for hidden layers and output
Layers: 6
Density: 100
Batch Size: 32

It stopped and it was here:
Epoch 19/100
30000/30000 ━━━━━━━━━━━━━━━━━━━━ 45s 1ms/step - loss: 1.1647 - mean_squared_logarithmic_error: 1.1647 - val_loss: 1.1435 - val_mean_squared_logarithmic_error: 1.1435 - learning_rate: 0.0956
But it took 24 epochs to finish. Comparing it to the last one it seems like its very close or even a little worse so let me predict and submit

Now its time that I do predict and see how good I did
I got 1.07487 which is slightly better than the last one but not by much

### THIRD TRY

model_file: model3.keras
plot_file: learning-curve3.png

I will go back to relu since maybe adding more layers would make it better.
Model: Sequential
Activation: relu for hidden layers and linear for output
Initializer = he_normal for hidden layers and output
Layers: 4
Density: 100
Batch Size: 32

It stopped and it was here:
Epoch 21/100
30000/30000 ━━━━━━━━━━━━━━━━━━━━ 38s 1ms/step - loss: 1.1648 - mean_squared_logarithmic_error: 1.1648 - val_loss: 1.1433 - val_mean_squared_logarithmic_error: 1.1433 - learning_rate: 0.0946
But it took 25 epochs to finish. Comparing it to the last one it is literally the same loss

Now its time that I do predict and see how good I did
I got 1.07428 which is kind of the same as the last one but it has 2 less layers.

