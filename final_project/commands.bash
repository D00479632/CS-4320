#!/bin/bash

# Plot the data to visualize it. (The label is harcoded in the arguments in display_data.py)
#./display_data.py all --data-file data/diabetes_dataset.csv

# Now I will split the data into train, validate and test.
#python3 split_data.py
# Original data shape: (100000, 17)
# Test data shape: (20000, 17)
# Training data shape final: (72000, 17)
# Validation data shape: (8000, 17)

# And now I will preprocess the data
#python3 preprocess.py
# Shape for data/preprocessed-diabetes-train.csv: (72000, 77)
# Shape for data/preprocessed-train.csv: (8000, 77)
# Shape for data/preprocessed-diabetes-test.csv: (20000, 77)