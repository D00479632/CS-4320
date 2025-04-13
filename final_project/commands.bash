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
# When doing show-test 1 make sure that you pass in the validate.csv not the actual test file
#./pipeline.py score --show-test 1 --model-type SGD --train-file data/diabetes_train.csv --test-file data/diabetes_validate.csv --model-file models/SGDClassifier.joblib
# diabetes_train: train_score: 0.9614166666666667 test_score: 0.959125
# I am not sure if the validate file is good or if validate is just a chunk of train that I didn't take out because this is very odd
# grep -f data/diabetes_validate.csv data/diabetes_test.csv
# ,year,gender,age,location,race:AfricanAmerican,race:Asian,race:Caucasian,race:Hispanic,race:Other,hypertension,heart_disease,smoking_history,bmi,hbA1c_level,blood_glucose_level,diabetes,clinical_notes
# With this command I know its good
# However, from the split data part that I did I can see that they are independent...
# I think we might have overfitted so lets check with the confusion matrix and the plots

#echo ==== CM Training Data ====
#./pipeline.py confusion-matrix --model-type SGD --train-file data/diabetes_train.csv --model-file models/SGDClassifier.joblib
#echo ==== CM Validation Data ====
#./pipeline.py confusion-matrix --model-type SGD --train-file data/diabetes_validate.csv --model-file models/SGDClassifier.joblib
#./pipeline.py precision-recall-plot --model-type SGD --train-file data/diabetes_train.csv --model-file models/SGDClassifier.joblib --image-file plots/SGDClassifier_pr_plot.png
#./pipeline.py pr-curve --model-type SGD --train-file data/diabetes_train.csv --model-file models/SGDClassifier.joblib --image-file plots/SGDClassifier_pr_curve.png

<<COMMENT
==== CM Training Data ====


     t/p      F     T 
        F 64608.0 1307.0 
        T 2232.0 3853.0 


Precision: 0.747
Recall:    0.633
F1:        0.685

==== CM Validation Data ====


     t/p      F     T 
        F 6875.0 413.0 
        T 217.0 495.0 


Precision: 0.545
Recall:    0.695
F1:        0.611

We can see severe overfitting (0.961 train vs ~0.74 test F1)
I also don't understand how the test_score with validate file was so high.
I am very confused with the results

Because there are so many non diabetic the model is bias.
We could change the data to make it more equal in between positive and negative by dropping some negative rows to make it equal with the positive.
If we don't change the data then we would need to have into account that a 95% accuracy is not great and that we need to bring to 0 the false positives.
COMMENT

# I know from past assignments that forest did better than the SGDClassifier so my next model will just be the random forest.
# I am also going to add the polynomial features to degree 2
#./pipeline.py random-search --model-type forest --train-file data/diabetes_train.csv --model-file models/RandomForestClassifier.joblib --search-grid-file models/SearchGridRandomForestClassifier.joblib --use-polynomial-features 2 --use-scaler 1 --categorical-missing-strategy most_frequent --numerical-missing-strategy median --n-search-iterations 10 

# Let's see what our best parameters are
#./pipeline.py show-best-params --model-type forest --train-file data/diabetes_train.csv --search-grid-file models/SearchGridRandomForestClassifier.joblib 
<<COMMENT
Best Score: 0.9715277777777778
Best Params:
{   'features__categorical__categorical-features-only__do_numerical': False,
    'features__categorical__categorical-features-only__do_predictors': True,
    'features__categorical__encode-category-bits__categories': 'auto',
    'features__categorical__encode-category-bits__handle_unknown': 'ignore',
    'features__categorical__missing-data__strategy': 'most_frequent',
    'features__numerical__missing-data__strategy': 'median',
    'features__numerical__numerical-features-only__do_numerical': True,
    'features__numerical__numerical-features-only__do_predictors': True,
    'features__numerical__polynomial-features__degree': 2,
    'model__bootstrap': True,
    'model__ccp_alpha': 0.0,
    'model__class_weight': None,
    'model__criterion': 'gini',
    'model__max_depth': None,
    'model__max_features': 'sqrt',
    'model__max_leaf_nodes': None,
    'model__max_samples': None,
    'model__min_impurity_decrease': 0.0,
    'model__min_samples_leaf': 1,
    'model__min_samples_split': 2,
    'model__min_weight_fraction_leaf': 0.0,
    'model__monotonic_cst': None,
    'model__n_estimators': 100,
    'model__n_jobs': None,
    'model__oob_score': False,
    'model__random_state': None,
    'model__verbose': 0,
    'model__warm_start': False}
COMMENT
# We might have over fitted


#./pipeline.py score --show-test 1 --model-type forest --train-file data/diabetes_train.csv --test-file data/diabetes_validate.csv --model-file models/RandomForestClassifier.joblib 
# diabetes_train: train_score: 0.9999722222222223 test_score: 0.9695
# However, looking at the validation score (the one that says test_score it's still pretty good)


#echo ==== CM Training Data ====
#./pipeline.py confusion-matrix --model-type forest --train-file data/diabetes_train.csv --model-file models/RandomForestClassifier.joblib 
<<COMMENT
==== CM Training Data ====


     t/p      F     T 
        F 65887.0  28.0 
        T 1990.0 4095.0 

Precision: 0.993
Recall:    0.673
F1:        0.802
COMMENT
#echo ==== CM Validation Data ====
#./pipeline.py confusion-matrix --model-type forest --train-file data/diabetes_validate.csv --model-file models/RandomForestClassifier.joblib
<<COMMENT
==== CM Validation Data ====


     t/p      F     T 
        F 7288.0   0.0 
        T 240.0 472.0 

Precision: 1.000
Recall:    0.663
F1:        0.797
COMMENT

# From the matrices I can see that the model is good at not identifying false positive cases. However, it's not that good when in the recall because it 
# identifies some false negatives so we are missing not good at predicting some positive cases which for this task is not good.

#./pipeline.py precision-recall-plot --model-type forest --train-file data/diabetes_train.csv --model-file models/RandomForestClassifier.joblib --image-file plots/RandomForestClassifier_pr_plot.png
#./pipeline.py pr-curve --model-type forest --train-file data/diabetes_train.csv --model-file models/RandomForestClassifier.joblib --image-file plots/RandomForestClassifier_pr_curve.png


# Since we noticed the unbalanced problem with the negative and positive cases and that was making the model bias I decided to clean the data and leave some of the 
# negative cases out so that it would be equal to the positive cases.

<<COMMENT
./clean_data.py
Reading data from data/diabetes_dataset.csv...
Original dataset:
Positive cases: 8500
Negative cases: 91500
As we can see here there are 91500 negative cases vs the 8500 positive which is not great

Balanced dataset:
Total cases: 17000
Positive cases: 8500
Negative cases: 8500
After the cleaning we end up with 17000 rows which is not as good as the original 100000 but I think that we can still work with it.
COMMENT

# Now we need to split the data again
#python3 split_data.py
<<COMMENT
Original data shape: (17000, 17)
Test data shape: (3400, 17)
Training data shape initial: (13600, 17)
Training data shape final: (10880, 17)
Validation data shape: (2720, 17)
COMMENT

# Even though is not as much training data as before I think that 10000 cases will do. 

# I won't make new models yet, I want to see how changing the data changed the outcome of the past models