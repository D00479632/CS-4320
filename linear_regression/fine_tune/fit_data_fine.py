#!/usr/bin/env python3

import sklearn
import sklearn.preprocessing
import sklearn.linear_model
import joblib

# read data, define fields, etc.
from showcase_common_fine import *

# peek at data
print(data.head(5))

# IMPORTANT: WE NEVER SCALE THE LABEL (Y)

# scale data with x' = (x - u) / s
# Subtracting the mean and dividing by the standard deviation
# Those two values are saved in the scaler object so next time we need to scale
# data (the testing data for example) we dont need to fit again we just transform
scaler = sklearn.preprocessing.StandardScaler()
# find u and s
scaler.fit(X_train) 
# transform data
X_train = scaler.transform(X_train) 

# peek at scaled data
print("Scaled Features")
print(feature_names)
print(X_train[:5,:])

# do the fit/training
# Max iterations changed so that the model gets a stabalized answer
regressor = sklearn.linear_model.SGDRegressor(max_iter=10000)
# TODO: Why do we fit the model again if we already had the fit data passed into it
regressor.fit(X_train, y_train)

# save the trained model
joblib.dump((regressor,scaler), model_filename)
