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
# Shape for data/preprocessed-diabetes-validate.csv: (8000, 77)
# Shape for data/preprocessed-diabetes-test.csv: (20000, 77)

# Let's do some grid searches with different models (SGDClassifier, RidgeClassifier, RandomForestClassifier, etc)
# for now lets not do --use-polynomial-features 2 maybe after
#./pipeline.py random-search --model-type SGD --train-file data/diabetes_train.csv --model-file models/SGDClassifier.joblib --search-grid-file models/SearchGridSGDClassifier.joblib  --use-scaler 1 --categorical-missing-strategy most_frequent --numerical-missing-strategy median --n-search-iterations 10 

# Let's see what our best parameters are
#./pipeline.py show-best-params --model-type SGD --train-file data/diabetes_train.csv --search-grid-file models/SearchGridSGDClassifier.joblib 

<<COMMENT
Fitting 3 folds for each of 10 candidates, totalling 30 fits
Best Score: 0.9609861111111111
Best Params:
{   'features__categorical__categorical-features-only__do_numerical': False,
    'features__categorical__categorical-features-only__do_predictors': True,
    'features__categorical__encode-category-bits__categories': 'auto',
    'features__categorical__encode-category-bits__handle_unknown': 'ignore',
    'features__categorical__missing-data__strategy': 'most_frequent',
    'features__numerical__missing-data__strategy': 'median',
    'features__numerical__numerical-features-only__do_numerical': True,
    'features__numerical__numerical-features-only__do_predictors': True,
    'model__alpha': 0.0001,
    'model__average': False,
    'model__class_weight': None,
    'model__early_stopping': True,
    'model__epsilon': 0.1,
    'model__eta0': 0.1,
    'model__fit_intercept': True,
    'model__l1_ratio': 0.5,
    'model__learning_rate': 'adaptive',
    'model__loss': 'hinge',
    'model__max_iter': 1000,
    'model__n_iter_no_change': 5,
    'model__n_jobs': None,
    'model__penalty': 'elasticnet',
    'model__power_t': 0.5,
    'model__random_state': None,
    'model__shuffle': True,
    'model__tol': 0.001,
    'model__validation_fraction': 0.1,
    'model__verbose': 0,
    'model__warm_start': False}
COMMENT

# Lets see the score of this model
#./pipeline.py score --model-type SGD --train-file data/diabetes_train.csv --test-file data/diabetes_test.csv --model-file models/SGDClassifier.joblib
# diabetes_train: train_score: 0.9614166666666667
# I think we might have overfitted so lets check with the confusion matrix and the plots

#./pipeline.py confusion-matrix --model-type SGD --train-file data/diabetes_train.csv --model-file models/SGDClassifier.joblib
#./pipeline.py precision-recall-plot --model-type SGD --train-file data/diabetes_train.csv --model-file models/SGDClassifier.joblib --image-file plots/SGDClassifier_pr_plot.png
#./pipeline.py pr-curve --model-type SGD --train-file data/diabetes_train.csv --model-file models/SGDClassifier.joblib --image-file plots/SGDClassifier_pr_curve.png

<<COMMENT
t/p      F     T 
    F 62453.0 3462.0 
    T 1684.0 4401.0 

Precision: 0.560
Recall:    0.723
F1:        0.631

We can see severe overfitting (0.961 train vs ~0.63 test F1)
COMMENT
