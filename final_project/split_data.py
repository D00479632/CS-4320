#!/usr/bin/env python3

import pandas as pd
import sklearn
import sklearn.model_selection

filename = "data/balanced_diabetes_dataset.csv"
train_filename = "data/balanced_train.csv"
test_filename = "data/balanced_test.csv"
validate_filename = "data/balanced_validate.csv"
# Pandas is smart enough to know that the first line is not data
data = pd.read_csv(filename)
# Output the shape of the original data
print("Original data shape:", data.shape)

# This is not a good seed and should be changed
# Once changed don't change it again so that if the program is rerun the split is the same
seed = 42

# This gives 20% of your data for testing
ratio = 0.2

# The \ is used bc the line is too long and that ensures continuity
data_train, data_test = \
    sklearn.model_selection.train_test_split(data, test_size=ratio, random_state=seed)

# Output the shape of the test data
print("Test data shape:", data_test.shape)

# Output the shape of the training data
print("Training data shape initial:", data_train.shape)

ratio = 0.2 
data_train, data_validate = \
    sklearn.model_selection.train_test_split(data_train, test_size=ratio, random_state=seed)

# Output the shape of the training data
print("Training data shape final:", data_train.shape)

# Output the shape of the validation data
print("Validation data shape:", data_validate.shape)

data_validate.to_csv(validate_filename)
data_train.to_csv(train_filename)
data_test.to_csv(test_filename)