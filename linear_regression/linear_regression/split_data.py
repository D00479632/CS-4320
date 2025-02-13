#!/usr/bin/env python3

import pandas as pd
import sklearn

filename = "csv/data.csv"
train_filename = "csv/training-data.csv"
test_filename = "csv/testing-data.csv"
# Pandas is smart enough to know that the first line is not data
data = pd.read_csv(filename)
# This is not a good seed and should be changed (seed 42)
# Once changed don't change it again so that if the program is rerun the split is the same
seed = 2468
# This gives 20% of your data for testing
ratio = 0.2
# The \ is used bc the line is too long and that ensures continuity
data_train, data_test = \
    sklearn.model_selection.train_test_split(data, test_size=ratio, random_state=seed)

data_train.to_csv(train_filename)
data_test.to_csv(test_filename)
