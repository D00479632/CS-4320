#!/usr/bin/env python3

import sklearn
import joblib

# read data, define fields, etc.
from global_variables import *

# The test_filename is not in the showcase_common because we want to make sure that 
# we dont accidentally use it. This is very precious data
test_filename = "csv/testing-data.csv"
test_data = pd.read_csv(test_filename, index_col=0)
X_test = test_data[feature_names]
y_test = test_data[label_name]

# load model
regressor = joblib.load(model_filename)

# ask model to score the data
score_test = regressor.score(X_test, y_test)
print("R^2: {}".format(score_test))

# show mean square error and mean absolute error
y_predicted = regressor.predict(X_test)
loss_test = sklearn.metrics.mean_squared_error(y_test, y_predicted)
print("MSE: {}".format(loss_test))
loss_test = sklearn.metrics.mean_absolute_error(y_test, y_predicted)
print("MAE: {}".format(loss_test))
