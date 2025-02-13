#!/usr/bin/env python3

import pandas as pd

###########################################################
# Set global values for filenames, features, label
# Load data
###########################################################

feature_names = ["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)"]

label_name = "Grades"
train_filename = "csv/training-data.csv"

# TODO: what does index_col=0 do to the data?
data = pd.read_csv(train_filename, index_col=0)
X_train = data[feature_names]
y_train = data[label_name]

model_filename = "model-linear-regressor.joblib"
