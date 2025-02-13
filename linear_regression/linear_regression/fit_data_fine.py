#!/usr/bin/env python3

import sklearn
import sklearn.preprocessing
import sklearn.linear_model
import joblib

# read data, define fields, etc.
from global_variables import *

# peek at data
print(data.head(5))

# IMPORTANT: WE NEVER SCALE THE LABEL (Y)

# scale data with x' = (x - u) / s
# Subtracting the mean and dividing by the standard deviation
# Those two values are saved in the scaler object so next time we need to scale
# data (the testing data for example) we dont need to fit again we just transform

# Model2 & Model3
#scaler = sklearn.preprocessing.StandardScaler()

# Model 4
scaler = sklearn.preprocessing.PolynomialFeatures(degree=2)

# find u and s
scaler.fit(X_train) 
# transform data
X_train = scaler.transform(X_train) 

# peek at scaled data
print("Scaled Features")
print(feature_names)
print(X_train[:5,:])

# Model2
#regressor = sklearn.linear_model.SGDRegressor(max_iter=10000)
# Model3
#regressor = sklearn.linear_model.SGDRegressor(loss='epsilon_insensitive', penalty='l1', max_iter=10000)
# Model4
regressor = sklearn.linear_model.BayesianRidge(max_iter=10000)
# TODO: Why do we fit the model again if we already had the fit data passed into it
regressor.fit(X_train, y_train)

# save the trained model
joblib.dump((regressor,scaler), model_filename)
