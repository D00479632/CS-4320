#!/usr/bin/env python3

import sklearn
import sklearn.linear_model
import joblib

# read data, define fields, etc.
from global_variables import *

# peek at data
print(data.head(5))

# do the fit/training there are other models you can use
# verbose=1 means the model will give you feedback
# Read the docs to see what error the model tries to minimize

# Model1
regressor = sklearn.linear_model.SGDRegressor()
regressor.fit(X_train, y_train)

# joblib dumps the contents of any object and then you can get them back
# save the trained model
joblib.dump(regressor, model_filename)
