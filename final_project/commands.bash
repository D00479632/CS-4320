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
#./pipeline.py random-search --model-type SGD --train-file data/dropped_train.csv --model-file models/dropped_SGDClassifier.joblib --search-grid-file models/dropped_SearchGridSGDClassifier.joblib  --use-scaler 1 --categorical-missing-strategy most_frequent --numerical-missing-strategy median --n-search-iterations 10 

# Let's see what our best parameters are
#./pipeline.py show-best-params --model-type SGD --train-file data/dropped_train.csv --search-grid-file models/dropped_SearchGridSGDClassifier.joblib 
<<COMMENT
Best Score: 0.8804230877963596
Best Params:
{   'features__categorical__categorical-features-only__do_numerical': False,
    'features__categorical__categorical-features-only__do_predictors': True,
    'features__categorical__encode-category-bits__categories': 'auto',
    'features__categorical__encode-category-bits__handle_unknown': 'ignore',
    'features__categorical__missing-data__strategy': 'most_frequent',
    'features__numerical__missing-data__strategy': 'median',
    'features__numerical__numerical-features-only__do_numerical': True,
    'features__numerical__numerical-features-only__do_predictors': True,
    'model__alpha': 0.01,
    'model__average': False,
    'model__class_weight': None,
    'model__early_stopping': True,
    'model__epsilon': 0.1,
    'model__eta0': 0.1,
    'model__fit_intercept': True,
    'model__l1_ratio': 0.3,
    'model__learning_rate': 'optimal',
    'model__loss': 'hinge',
    'model__max_iter': 1000,
    'model__n_iter_no_change': 10,
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
# I already see a difference in the best score that before was 0.96098 and now is 0.880 the learning rate is now optimal and the n_iter_no_change is 10 

# Lets see the score of this model
# When doing show-test 1 make sure that you pass in the validate.csv not the actual test file
#./pipeline.py score --show-test 1 --model-type SGD --train-file data/dropped_train.csv --test-file data/dropped_validate.csv --model-file models/dropped_SGDClassifier.joblib
# dropped_train: train_score: 0.8809742647058824 test_score: 0.8900735294117647

#echo ==== CM Training Data ====
#./pipeline.py confusion-matrix --model-type SGD --train-file data/dropped_train.csv --model-file models/dropped_SGDClassifier.joblib
#echo ==== CM Validation Data ====
#./pipeline.py confusion-matrix --model-type SGD --train-file data/dropped_validate.csv --model-file models/dropped_SGDClassifier.joblib
#./pipeline.py precision-recall-plot --model-type SGD --train-file data/dropped_train.csv --model-file models/dropped_SGDClassifier.joblib --image-file plots/dropped_SGDClassifier_pr_plot.png
#./pipeline.py pr-curve --model-type SGD --train-file data/dropped_train.csv --model-file models/dropped_SGDClassifier.joblib --image-file plots/dropped_SGDClassifier_pr_curve.png
<<COMMENT
==== CM Training Data ====
     t/p      F     T 
        F 4624.0 792.0 
        T 761.0 4703.0 

Precision: 0.856
Recall:    0.861
F1:        0.858
==== CM Validation Data ====
     t/p      F     T 
        F 1140.0 242.0 
        T 223.0 1115.0 

Precision: 0.822
Recall:    0.833
F1:        0.827

We can already see improvement in the balance in between predicting positives and negatives, this score is way better but we are still missing like a 14% of true and false cases 
COMMENT

# Now let's try with the forest
# I just figured out that last time when I did the forest search in make_pipeline I didnt have multiple choices for the hyperparameters so the models wont be the same
#./pipeline.py random-search --model-type forest --train-file data/dropped_train.csv --model-file models/dropped_RandomForestClassifier.joblib --search-grid-file models/dropped_SearchGridRandomForestClassifier.joblib --use-polynomial-features 2 --use-scaler 1 --categorical-missing-strategy most_frequent --numerical-missing-strategy median --n-search-iterations 10 

# Let's see what our best parameters are
#./pipeline.py show-best-params --model-type forest --train-file data/dropped_train.csv --search-grid-file models/dropped_SearchGridRandomForestClassifier.joblib 

<<COMMENT
Best Score: 0.8966917745719588
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
    'model__bootstrap': False,
    'model__ccp_alpha': 0.01,
    'model__class_weight': None,
    'model__criterion': 'gini',
    'model__max_depth': 30,
    'model__max_features': 'log2',
    'model__max_leaf_nodes': None,
    'model__max_samples': None,
    'model__min_impurity_decrease': 0.0,
    'model__min_samples_leaf': 2,
    'model__min_samples_split': 5,
    'model__min_weight_fraction_leaf': 0.0,
    'model__monotonic_cst': None,
    'model__n_estimators': 100,
    'model__n_jobs': None,
    'model__oob_score': False,
    'model__random_state': None,
    'model__verbose': 0,
    'model__warm_start': False}
COMMENT

# Lets see the score of this model
#./pipeline.py score --show-test 1 --model-type forest --train-file data/dropped_train.csv --test-file data/dropped_validate.csv --model-file models/dropped_RandomForestClassifier.joblib
# dropped_train: train_score: 0.8991727941176471 test_score: 0.8985294117647059

#echo ==== CM Training Data ====
#./pipeline.py confusion-matrix --model-type forest --train-file data/dropped_train.csv --model-file models/dropped_RandomForestClassifier.joblib
#echo ==== CM Validation Data ====
#./pipeline.py confusion-matrix --model-type forest --train-file data/dropped_validate.csv --model-file models/dropped_RandomForestClassifier.joblib
#./pipeline.py precision-recall-plot --model-type forest --train-file data/dropped_train.csv --model-file models/dropped_RandomForestClassifier.joblib --image-file plots/dropped_RandomForestClassifier_pr_plot.png
#./pipeline.py pr-curve --model-type forest --train-file data/dropped_train.csv --model-file models/dropped_RandomForestClassifier.joblib --image-file plots/dropped_RandomForestClassifier_pr_curve.png
<<COMMENT
==== CM Training Data ====
     t/p      F     T 
        F 4868.0 548.0 
        T 534.0 4930.0 

Precision: 0.900
Recall:    0.902
F1:        0.901
==== CM Validation Data ====
     t/p      F     T 
        F 1258.0 124.0 
        T 126.0 1212.0 

Precision: 0.907
Recall:    0.906
F1:        0.907

This is way better than the other two models and than the SGD regressor, changing the data was a good thing we are not predicting as many negatives anymore
COMMENT

# I changed the score in the sklearn.model_selection.RandomizedSearchCV to be acuracy.
# Now I will try a linear model (RidgeClassifierCV)

#./pipeline.py random-search --model-type linear --train-file data/dropped_train.csv --model-file models/dropped_RidgeClassifierCV.joblib --search-grid-file models/dropped_SearchGridRidgeClassifierCV.joblib --use-polynomial-features 2 --use-scaler 1 --categorical-missing-strategy most_frequent --numerical-missing-strategy median --n-search-iterations 10 

# Let's see what our best parameters are
#./pipeline.py show-best-params --model-type linear --train-file data/dropped_train.csv --search-grid-file models/dropped_SearchGridRidgeClassifierCV.joblib 
<<COMMENT
Best Score: 0.8781250232863136
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
    'model__alphas': [0.01, 0.1, 1.0, 10.0, 100.0],
    'model__class_weight': None,
    'model__cv': 5,
    'model__fit_intercept': False,
    'model__scoring': None}
COMMENT

#./pipeline.py score --show-test 1 --model-type linear --train-file data/dropped_train.csv --test-file data/dropped_validate.csv --model-file models/dropped_RidgeClassifierCV.joblib
# dropped_train: train_score: 0.8805147058823529 test_score: 0.8878676470588235

#echo ==== CM Training Data ====
#./pipeline.py confusion-matrix --model-type linear --train-file data/dropped_train.csv --model-file models/dropped_RidgeClassifierCV.joblib
#echo ==== CM Validation Data ====
#./pipeline.py confusion-matrix --model-type linear --train-file data/dropped_validate.csv --model-file models/dropped_RidgeClassifierCV.joblib
#./pipeline.py precision-recall-plot --model-type linear --train-file data/dropped_train.csv --model-file models/dropped_RidgeClassifierCV.joblib --image-file plots/dropped_RidgeClassifierCV_pr_plot.png
#./pipeline.py pr-curve --model-type linear --train-file data/dropped_train.csv --model-file models/dropped_RidgeClassifierCV.joblib --image-file plots/dropped_RidgeClassifierCV_pr_curve.png

<<COMMENT 
==== CM Training Data ====
     t/p      F     T 
        F 4832.0 584.0 
        T 743.0 4721.0 

Precision: 0.890
Recall:    0.864
F1:        0.877
==== CM Validation Data ====
     t/p      F     T 
        F 1231.0 151.0 
        T 171.0 1167.0 

Precision: 0.885
Recall:    0.872
F1:        0.879
This is no better than the forest model so for now my best model is forest
COMMENT

# TODO: ask Curtis what ways I can improve the false negatives (when the model predicts negative but its positive), I rather tell non-diabetic people that they are diabetic than diabetic people that they arent
# kind of like covid test that have more false positives to make sure no false negative is present

# I changed the score in the grid search to be precision so that I can focus on the false negatives and true positives

#./pipeline.py random-search --model-type forest --train-file data/dropped_train.csv --model-file models/dropped_RandomForestClassifier2.joblib --search-grid-file models/dropped_SearchGridRandomForestClassifier2.joblib --use-polynomial-features 2 --use-scaler 1 --categorical-missing-strategy most_frequent --numerical-missing-strategy median --n-search-iterations 20 

# Let's see what our best parameters are
#./pipeline.py show-best-params --model-type forest --train-file data/dropped_train.csv --search-grid-file models/dropped_SearchGridRandomForestClassifier2.joblib 
<<COMMENT
Best Score: 0.9659927497110411
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
    'model__bootstrap': False,
    'model__ccp_alpha': 0.1,
    'model__class_weight': None,
    'model__criterion': 'entropy',
    'model__max_depth': 10,
    'model__max_features': 'sqrt',
    'model__max_leaf_nodes': None,
    'model__max_samples': None,
    'model__min_impurity_decrease': 0.2,
    'model__min_samples_leaf': 1,
    'model__min_samples_split': 10,
    'model__min_weight_fraction_leaf': 0.0,
    'model__monotonic_cst': None,
    'model__n_estimators': 300,
    'model__n_jobs': -1,
    'model__oob_score': False,
    'model__random_state': None,
    'model__verbose': 0,
    'model__warm_start': False}
COMMENT

# Lets see the score of this model
#./pipeline.py score --show-test 1 --model-type forest --train-file data/dropped_train.csv --test-file data/dropped_validate.csv --model-file models/dropped_RandomForestClassifier2.joblib
# dropped_train: train_score: 0.8421875 test_score: 0.8371323529411765

#echo ==== CM Training Data ====
#./pipeline.py confusion-matrix --model-type forest --train-file data/dropped_train.csv --model-file models/dropped_RandomForestClassifier2.joblib
#echo ==== CM Validation Data ====
#./pipeline.py confusion-matrix --model-type forest --train-file data/dropped_validate.csv --model-file models/dropped_RandomForestClassifier2.joblib
#./pipeline.py precision-recall-plot --model-type forest --train-file data/dropped_train.csv --model-file models/dropped_RandomForestClassifier2.joblib --image-file plots/dropped_RandomForestClassifier_pr_plot2.png
#./pipeline.py pr-curve --model-type forest --train-file data/dropped_train.csv --model-file models/dropped_RandomForestClassifier2.joblib --image-file plots/dropped_RandomForestClassifier_pr_curve2.png
<<COMMENT
NEW ONE:
==== CM Training Data ====


     t/p      F     T 
        F 4852.0 564.0 
        T 532.0 4932.0 


Precision: 0.897
Recall:    0.903
F1:        0.900
==== CM Validation Data ====


     t/p      F     T 
        F 1267.0 115.0 
        T 129.0 1209.0 


Precision: 0.913
Recall:    0.904
F1:        0.908

OLD ONE:
==== CM Training Data ====
     t/p      F     T 
        F 4868.0 548.0 
        T 534.0 4930.0 

Precision: 0.900
Recall:    0.902
F1:        0.901
==== CM Validation Data ====
     t/p      F     T 
        F 1258.0 124.0 
        T 126.0 1212.0 

Precision: 0.907
Recall:    0.906
F1:        0.907


There is not that much of a change so I think that my old model is still the best one, now I will move on to the neural networks
COMMENT

# Curtis recommended to run it again but give it 300 trials to go and see if that changes anythoing
# I am going to just keep the same namefiles so they will be overwritten

#./pipeline.py random-search --model-type forest --train-file data/dropped_train.csv --model-file models/dropped_RandomForestClassifier2.joblib --search-grid-file models/dropped_SearchGridRandomForestClassifier2.joblib --use-polynomial-features 2 --use-scaler 1 --categorical-missing-strategy most_frequent --numerical-missing-strategy median --n-search-iterations 300 

# Let's see what our best parameters are
#./pipeline.py show-best-params --model-type forest --train-file data/dropped_train.csv --search-grid-file models/dropped_SearchGridRandomForestClassifier2.joblib 

<<COMMENT
Best Score: 0.9753511955970132
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
    'model__bootstrap': False,
    'model__ccp_alpha': 0.1,
    'model__class_weight': 'balanced',
    'model__criterion': 'entropy',
    'model__max_depth': 10,
    'model__max_features': 'log2',
    'model__max_leaf_nodes': None,
    'model__max_samples': None,
    'model__min_impurity_decrease': 0.2,
    'model__min_samples_leaf': 2,
    'model__min_samples_split': 2,
    'model__min_weight_fraction_leaf': 0.0,
    'model__monotonic_cst': None,
    'model__n_estimators': 300,
    'model__n_jobs': -1,
    'model__oob_score': False,
    'model__random_state': None,
    'model__verbose': 0,
    'model__warm_start': False}
COMMENT

# Lets see the score of this model
#./pipeline.py score --show-test 1 --model-type forest --train-file data/dropped_train.csv --test-file data/dropped_validate.csv --model-file models/dropped_RandomForestClassifier2.joblib
# dropped_train: train_score: 0.8431985294117647 test_score: 0.8356617647058824

#echo ==== CM Training Data ====
#./pipeline.py confusion-matrix --model-type forest --train-file data/dropped_train.csv --model-file models/dropped_RandomForestClassifier2.joblib
#echo ==== CM Validation Data ====
#./pipeline.py confusion-matrix --model-type forest --train-file data/dropped_validate.csv --model-file models/dropped_RandomForestClassifier2.joblib
#./pipeline.py precision-recall-plot --model-type forest --train-file data/dropped_train.csv --model-file models/dropped_RandomForestClassifier2.joblib --image-file plots/dropped_RandomForestClassifier_pr_plot2.png
#./pipeline.py pr-curve --model-type forest --train-file data/dropped_train.csv --model-file models/dropped_RandomForestClassifier2.joblib --image-file plots/dropped_RandomForestClassifier_pr_curve2.png

<<COMMENT
==== CM Training Data ====
     t/p      F     T 
        F 4851.0 565.0 
        T 532.0 4932.0 

Precision: 0.897
Recall:    0.903
F1:        0.900
==== CM Validation Data ====
     t/p      F     T 
        F 1255.0 127.0 
        T 129.0 1209.0 

Precision: 0.905
Recall:    0.904
F1:        0.904
COMMENT

#./pipeline.py random-search --model-type ada --train-file data/dropped_train.csv --model-file models/dropped_AdaBoostClassifier.joblib --search-grid-file models/dropped_SearchGridAdaBoostClassifier.joblib --use-polynomial-features 2 --use-scaler 1 --categorical-missing-strategy most_frequent --numerical-missing-strategy median --n-search-iterations 300 

# Let's see what our best parameters are
#./pipeline.py show-best-params --model-type ada --train-file data/dropped_train.csv --search-grid-file models/dropped_SearchGridAdaBoostClassifier.joblib 
<<COMMENT
Best Score: 0.8955556402212524
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
    'model__learning_rate': 0.1,
    'model__n_estimators': 100}
COMMENT

# Lets see the score of this model
#./pipeline.py score --show-test 1 --model-type ada --train-file data/dropped_train.csv --test-file data/dropped_validate.csv --model-file models/dropped_AdaBoostClassifier.joblib
# dropped_train: train_score: 0.8897058823529411 test_score: 0.8856617647058823

#echo ==== CM Training Data ====
#./pipeline.py confusion-matrix --model-type ada --train-file data/dropped_train.csv --model-file models/dropped_AdaBoostClassifier.joblib
#echo ==== CM Validation Data ====
#./pipeline.py confusion-matrix --model-type ada --train-file data/dropped_validate.csv --model-file models/dropped_AdaBoostClassifier.joblib
#./pipeline.py precision-recall-plot --model-type ada --train-file data/dropped_train.csv --model-file models/dropped_AdaBoostClassifier.joblib --image-file plots/dropped_AdaBoostClassifier_pr_plot2.png
#./pipeline.py pr-curve --model-type ada --train-file data/dropped_train.csv --model-file models/dropped_AdaBoostClassifier.joblib --image-file plots/dropped_AdaBoostClassifier_pr_curve2.png
<<COMMENT
==== CM Training Data ====
     t/p      F     T 
        F 4763.0 653.0 
        T 471.0 4993.0 

Precision: 0.884
Recall:    0.914
F1:        0.899
==== CM Validation Data ====
     t/p      F     T 
        F 1249.0 133.0 
        T 121.0 1217.0 

Precision: 0.901
Recall:    0.910
F1:        0.906
COMMENT

# I can actually see improvement with this model, there are more false positives but less false negatives so for me that is an advancement.

# Now lets do neural networks to see how it does

<<COMMENT
model_name=a

echo "=== Starting CNN training process ==="
echo "Model name: ${model_name}"

echo "[1/1] Fitting initial model with all the data..."
time ./cnn_classification.py cnn-fit \
     --model-name ${model_name} --model-file models/${model_name}.joblib \
     --batch-number 1

echo "[1/1] Generating learning curve..."
time ./cnn_classification.py learning-curve \
     --model-file models/${model_name}.joblib --learning-curve-file plots/${model_name}.learning_curve.png

echo "Generating score"
time ./cnn_classification.py score \
     --model-file models/${model_name}.joblib \
     --batch-number 2

=== Starting CNN training process ===
Model name: diabetes
[1/1] Fitting initial model with all the data...
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 64)                  │           4,992 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 32)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 2)                   │              66 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 7,138 (27.88 KB)
 Trainable params: 7,138 (27.88 KB)
 Non-trainable params: 0 (0.00 B)
None
Epoch 1/50
2025-04-20 17:57:19.386341: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.
544/544 ━━━━━━━━━━━━━━━━━━━━ 9s 11ms/step - loss: 466.5778 - precision: 0.4973 - val_loss: 102.7866 - val_precision: 0.5018
Epoch 2/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - loss: 213.5778 - precision: 0.5242 - val_loss: 8.4700 - val_precision: 0.5988
Epoch 3/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - loss: 112.8842 - precision: 0.5202 - val_loss: 11.1033 - val_precision: 0.5685
Epoch 4/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - loss: 47.7494 - precision: 0.5369 - val_loss: 0.4561 - val_precision: 0.8438
Epoch 5/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - loss: 23.1884 - precision: 0.5635 - val_loss: 7.1105 - val_precision: 0.5597
Epoch 6/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - loss: 17.6721 - precision: 0.5573 - val_loss: 1.0839 - val_precision: 0.6507
Epoch 7/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - loss: 11.7128 - precision: 0.5481 - val_loss: 8.2441 - val_precision: 0.5225
Epoch 8/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - loss: 9.0351 - precision: 0.5215 - val_loss: 11.8835 - val_precision: 0.5115
Epoch 9/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - loss: 7.1652 - precision: 0.5050 - val_loss: 1.1605 - val_precision: 0.5299

[1/1] Generating learning curve...
GPU is available

Generating score
GPU is available
85/85 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step 

models/diabetes1.joblib: train: 

+-----+-----+
|1198 | 184 |
| 207 |1131 |
+-----+-----+

If we compare this confusion matrix with my best model:
==== CM Validation Data ====
     t/p      F     T 
        F 1249.0 133.0 
        T 121.0 1217.0 

we are predicting a lot more false negatives which is not good and not what I want

precision: 0.8600760456273764
recall: 0.8452914798206278
f1: 0.8526196758386732


real    0m3.241s
user    0m4.673s
sys     0m0.313s


I ran the same thing again and got something completely different:
=== Starting CNN training process ===
Model name: a
[1/1] Fitting initial model with all the data...
GPU is available
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 64)                  │           4,992 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 32)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 2)                   │              66 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 7,138 (27.88 KB)
 Trainable params: 7,138 (27.88 KB)
 Non-trainable params: 0 (0.00 B)
None
Epoch 1/50
2025-04-20 18:07:41.133859: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.
544/544 ━━━━━━━━━━━━━━━━━━━━ 7s 11ms/step - loss: 406.4063 - precision: 0.5100 - val_loss: 25.3938 - val_precision: 0.5202
Epoch 2/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - loss: 191.7362 - precision: 0.5290 - val_loss: 3.1168 - val_precision: 0.5823
Epoch 3/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - loss: 88.5556 - precision: 0.5332 - val_loss: 0.6495 - val_precision: 0.7647
Epoch 4/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - loss: 28.7609 - precision: 0.5375 - val_loss: 12.6152 - val_precision: 0.5198
Epoch 5/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - loss: 17.5471 - precision: 0.5419 - val_loss: 8.6038 - val_precision: 0.5193
Epoch 6/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - loss: 9.5538 - precision: 0.5199 - val_loss: 6.9664 - val_precision: 0.5142
Epoch 7/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - loss: 5.5219 - precision: 0.5369 - val_loss: 21.6714 - val_precision: 0.4940
Epoch 8/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - loss: 5.1507 - precision: 0.5213 - val_loss: 13.5184 - val_precision: 0.5124

[1/1] Generating learning curve...
GPU is available

Generating score
GPU is available
85/85 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step 

models/a.joblib: train: 

+-----+-----+
| 757 | 625 |
|  14 |1324 |
+-----+-----+

precision: 0.6793227296049256
recall: 0.9895366218236173
f1: 0.8055978095527837


real    0m3.240s
user    0m4.643s
sys     0m0.350s

The precision got way worse but the recall is great
COMMENT

<<COMMENT
model_name=b

echo "=== Starting CNN training process ==="
echo "Model name: ${model_name}"

echo "[1/1] Fitting initial model with all the data..."
time ./cnn_classification.py cnn-fit \
     --model-name ${model_name} --model-file models/${model_name}.joblib \
     --batch-number 1

echo "[1/1] Generating learning curve..."
time ./cnn_classification.py learning-curve \
     --model-file models/${model_name}.joblib --learning-curve-file plots/${model_name}_learning_curve.png

echo "Generating training score"
time ./cnn_classification.py score \
     --model-file models/${model_name}.joblib \
     --batch-number 1

echo "Generating validation score"
time ./cnn_classification.py score \
     --model-file models/${model_name}.joblib \
     --batch-number 2

=== Starting CNN training process ===
Model name: b
[1/1] Fitting initial model with all the data...
GPU is available
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 128)                 │           9,984 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, 128)                 │             512 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 64)                  │           8,256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_1                │ (None, 64)                  │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_2                │ (None, 32)                  │             128 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 2)                   │              66 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 21,282 (83.13 KB)
 Trainable params: 20,834 (81.38 KB)
 Non-trainable params: 448 (1.75 KB)
None
Epoch 1/50
2025-04-20 18:22:22.065166: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.
544/544 ━━━━━━━━━━━━━━━━━━━━ 13s 20ms/step - Recall: 0.4934 - loss: 0.8594 - val_Recall: 0.5041 - val_loss: 0.7058
Epoch 2/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 20ms/step - Recall: 0.5042 - loss: 0.7116 - val_Recall: 0.5032 - val_loss: 0.7262
Epoch 3/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - Recall: 0.4968 - loss: 0.7132 - val_Recall: 0.5083 - val_loss: 0.6920
Epoch 4/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.5159 - loss: 0.6976 - val_Recall: 0.5142 - val_loss: 0.6937
Epoch 5/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.5048 - loss: 0.6986 - val_Recall: 0.4963 - val_loss: 0.7053
Epoch 6/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.5113 - loss: 0.6967 - val_Recall: 0.5055 - val_loss: 0.6935
Epoch 7/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.5136 - loss: 0.6990 - val_Recall: 0.5055 - val_loss: 0.6980
Epoch 8/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.5156 - loss: 0.6975 - val_Recall: 0.5786 - val_loss: 0.6889
Epoch 9/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.5132 - loss: 0.6974 - val_Recall: 0.5207 - val_loss: 0.7002
Epoch 10/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.5146 - loss: 0.6992 - val_Recall: 0.5519 - val_loss: 0.6873
Epoch 11/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.5385 - loss: 0.6923 - val_Recall: 0.5345 - val_loss: 0.6942
Epoch 12/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.5552 - loss: 0.6916 - val_Recall: 0.5290 - val_loss: 0.7185
Epoch 13/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - Recall: 0.5501 - loss: 0.6935 - val_Recall: 0.5666 - val_loss: 0.6824
Epoch 14/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - Recall: 0.5686 - loss: 0.6859 - val_Recall: 0.5083 - val_loss: 0.7235
Epoch 15/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.6000 - loss: 0.6717 - val_Recall: 0.5630 - val_loss: 0.6712
Epoch 16/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - Recall: 0.5795 - loss: 0.6860 - val_Recall: 0.5680 - val_loss: 0.6717
Epoch 17/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - Recall: 0.5830 - loss: 0.6827 - val_Recall: 0.5028 - val_loss: 0.6714
Epoch 18/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.5878 - loss: 0.6815 - val_Recall: 0.5800 - val_loss: 0.6657
Epoch 19/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 20ms/step - Recall: 0.5837 - loss: 0.6800 - val_Recall: 0.8199 - val_loss: 0.6581
Epoch 20/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.6026 - loss: 0.6740 - val_Recall: 0.6415 - val_loss: 0.6590
Epoch 21/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.6096 - loss: 0.6709 - val_Recall: 0.5634 - val_loss: 0.6650
Epoch 22/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.5777 - loss: 0.6820 - val_Recall: 0.6209 - val_loss: 0.6574
Epoch 23/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.5885 - loss: 0.6811 - val_Recall: 0.7100 - val_loss: 0.6536
Epoch 24/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - Recall: 0.5985 - loss: 0.6760 - val_Recall: 0.6016 - val_loss: 0.6359
Epoch 25/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - Recall: 0.6192 - loss: 0.6631 - val_Recall: 0.5915 - val_loss: 0.6675
Epoch 26/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - Recall: 0.6142 - loss: 0.6709 - val_Recall: 0.6466 - val_loss: 0.6463
Epoch 27/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - Recall: 0.6263 - loss: 0.6651 - val_Recall: 0.8254 - val_loss: 0.6217
Epoch 28/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - Recall: 0.6649 - loss: 0.6333 - val_Recall: 0.6498 - val_loss: 0.6217
Epoch 29/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - Recall: 0.6519 - loss: 0.6443 - val_Recall: 0.6673 - val_loss: 0.6006
Epoch 30/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.6715 - loss: 0.6303 - val_Recall: 0.6696 - val_loss: 0.5753
Epoch 31/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 20ms/step - Recall: 0.6938 - loss: 0.5988 - val_Recall: 0.7803 - val_loss: 0.5006
Epoch 32/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.7206 - loss: 0.5581 - val_Recall: 0.6314 - val_loss: 0.7345
Epoch 33/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - Recall: 0.7130 - loss: 0.5565 - val_Recall: 0.6760 - val_loss: 0.5499
Epoch 34/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.7382 - loss: 0.5371 - val_Recall: 0.8318 - val_loss: 0.4164
Epoch 35/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - Recall: 0.7498 - loss: 0.5018 - val_Recall: 0.7537 - val_loss: 0.4860
Epoch 36/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.7428 - loss: 0.5088 - val_Recall: 0.7509 - val_loss: 0.4806
Epoch 37/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.7358 - loss: 0.5243 - val_Recall: 0.7031 - val_loss: 0.5591
Epoch 38/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - Recall: 0.7376 - loss: 0.5232 - val_Recall: 0.7863 - val_loss: 0.4761
Epoch 39/50
544/544 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - Recall: 0.7200 - loss: 0.5573 - val_Recall: 0.8318 - val_loss: 0.4590

real    6m56.877s
user    7m17.311s
sys     2m52.701s
[1/1] Generating learning curve...
GPU is available

Generating training score
GPU is available
340/340 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step  

models/b.joblib: train: 

+-----+-----+
|3716 |1700 |
| 187 |5277 |
+-----+-----+

precision: 0.7563422674501935
recall: 0.9657759882869692
f1: 0.8483240897034


Generating validation score
GPU is available
85/85 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step 

models/b.joblib: train: 

+-----+-----+
| 975 | 407 |
|  38 |1300 |
+-----+-----+

precision: 0.7615700058582309
recall: 0.9715994020926756
f1: 0.8538587848932676


real    0m3.370s
user    0m4.810s
sys     0m0.388s
COMMENT

<<COMMENT
model_name=c

echo "=== Starting CNN training process ==="
echo "Model name: ${model_name}"

echo "[1/1] Fitting initial model with all the data..."
time ./cnn_classification.py cnn-fit \
     --model-name ${model_name} --model-file models/${model_name}.joblib \
     --batch-number 1

echo "[1/1] Generating learning curve..."
time ./cnn_classification.py learning-curve \
     --model-file models/${model_name}.joblib --learning-curve-file plots/${model_name}_learning_curve.png

echo "Generating training score"
time ./cnn_classification.py score \
     --model-file models/${model_name}.joblib \
     --batch-number 1

echo "Generating validation score"
time ./cnn_classification.py score \
     --model-file models/${model_name}.joblib \
     --batch-number 2

=== Starting CNN training process ===
Model name: c
[1/1] Fitting initial model with all the data...
GPU is available
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 128)                 │           9,984 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, 128)                 │             512 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 64)                  │           8,256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_1                │ (None, 64)                  │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_2                │ (None, 32)                  │             128 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 2)                   │              66 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 21,282 (83.13 KB)
 Trainable params: 20,834 (81.38 KB)
 Non-trainable params: 448 (1.75 KB)
None
Epoch 1/100
2025-04-20 19:16:42.137593: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.
272/272 ━━━━━━━━━━━━━━━━━━━━ 7s 21ms/step - Precision: 0.5149 - Recall: 0.5149 - loss: 0.8459 - val_Precision: 0.5005 - val_Recall: 0.5005 - val_loss: 0.7098
Epoch 2/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5218 - Recall: 0.5218 - loss: 0.7202 - val_Precision: 0.4839 - val_Recall: 0.4839 - val_loss: 0.7169
Epoch 3/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 21ms/step - Precision: 0.5184 - Recall: 0.5184 - loss: 0.7064 - val_Precision: 0.5188 - val_Recall: 0.5188 - val_loss: 0.6910
Epoch 4/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5275 - Recall: 0.5275 - loss: 0.7010 - val_Precision: 0.4839 - val_Recall: 0.4839 - val_loss: 0.7068
Epoch 5/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5244 - Recall: 0.5244 - loss: 0.6998 - val_Precision: 0.5335 - val_Recall: 0.5335 - val_loss: 0.6915
Epoch 6/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5087 - Recall: 0.5087 - loss: 0.7079 - val_Precision: 0.5391 - val_Recall: 0.5391 - val_loss: 0.6883
Epoch 7/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5179 - Recall: 0.5179 - loss: 0.6969 - val_Precision: 0.5685 - val_Recall: 0.5685 - val_loss: 0.6880
Epoch 8/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5208 - Recall: 0.5208 - loss: 0.6952 - val_Precision: 0.5023 - val_Recall: 0.5023 - val_loss: 0.6968
Epoch 9/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5387 - Recall: 0.5387 - loss: 0.6937 - val_Precision: 0.5446 - val_Recall: 0.5446 - val_loss: 0.6879
Epoch 10/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5523 - Recall: 0.5523 - loss: 0.6927 - val_Precision: 0.5455 - val_Recall: 0.5455 - val_loss: 0.6886
Epoch 11/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5359 - Recall: 0.5359 - loss: 0.6979 - val_Precision: 0.5083 - val_Recall: 0.5083 - val_loss: 0.6968
Epoch 12/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5360 - Recall: 0.5360 - loss: 0.6976 - val_Precision: 0.5165 - val_Recall: 0.5165 - val_loss: 0.6910
Epoch 13/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.5341 - Recall: 0.5341 - loss: 0.6935 - val_Precision: 0.5772 - val_Recall: 0.5772 - val_loss: 0.6853
Epoch 14/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5374 - Recall: 0.5374 - loss: 0.6950 - val_Precision: 0.5492 - val_Recall: 0.5492 - val_loss: 0.6856
Epoch 15/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5575 - Recall: 0.5575 - loss: 0.6925 - val_Precision: 0.5193 - val_Recall: 0.5193 - val_loss: 0.6980
Epoch 16/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.5603 - Recall: 0.5603 - loss: 0.6917 - val_Precision: 0.5519 - val_Recall: 0.5519 - val_loss: 0.6877
Epoch 17/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5384 - Recall: 0.5384 - loss: 0.6956 - val_Precision: 0.5124 - val_Recall: 0.5124 - val_loss: 0.6848
Epoch 18/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5464 - Recall: 0.5464 - loss: 0.6927 - val_Precision: 0.5271 - val_Recall: 0.5271 - val_loss: 0.6874
Epoch 19/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5366 - Recall: 0.5366 - loss: 0.6938 - val_Precision: 0.5303 - val_Recall: 0.5303 - val_loss: 0.7136
Epoch 20/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5592 - Recall: 0.5592 - loss: 0.6896 - val_Precision: 0.7068 - val_Recall: 0.7068 - val_loss: 0.6783
Epoch 21/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5464 - Recall: 0.5464 - loss: 0.6909 - val_Precision: 0.5188 - val_Recall: 0.5188 - val_loss: 0.6904
Epoch 22/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5534 - Recall: 0.5534 - loss: 0.6902 - val_Precision: 0.5312 - val_Recall: 0.5312 - val_loss: 0.6814
Epoch 23/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.5716 - Recall: 0.5716 - loss: 0.6867 - val_Precision: 0.5188 - val_Recall: 0.5188 - val_loss: 0.7291
Epoch 24/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.5512 - Recall: 0.5512 - loss: 0.6911 - val_Precision: 0.5579 - val_Recall: 0.5579 - val_loss: 0.6815
Epoch 25/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5811 - Recall: 0.5811 - loss: 0.6850 - val_Precision: 0.7275 - val_Recall: 0.7275 - val_loss: 0.6731
Epoch 26/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 21ms/step - Precision: 0.5900 - Recall: 0.5900 - loss: 0.6801 - val_Precision: 0.6075 - val_Recall: 0.6075 - val_loss: 0.6735
Epoch 27/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5742 - Recall: 0.5742 - loss: 0.6853 - val_Precision: 0.5960 - val_Recall: 0.5960 - val_loss: 0.6815
Epoch 28/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.5661 - Recall: 0.5661 - loss: 0.6890 - val_Precision: 0.5083 - val_Recall: 0.5083 - val_loss: 0.6696
Epoch 29/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.6060 - Recall: 0.6060 - loss: 0.6772 - val_Precision: 0.5689 - val_Recall: 0.5689 - val_loss: 0.6748
Epoch 30/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.6043 - Recall: 0.6043 - loss: 0.6786 - val_Precision: 0.6195 - val_Recall: 0.6195 - val_loss: 0.6733
Epoch 31/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5979 - Recall: 0.5979 - loss: 0.6776 - val_Precision: 0.5427 - val_Recall: 0.5427 - val_loss: 0.6639
Epoch 32/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.5966 - Recall: 0.5966 - loss: 0.6765 - val_Precision: 0.5648 - val_Recall: 0.5648 - val_loss: 0.6696
Epoch 33/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.5997 - Recall: 0.5997 - loss: 0.6794 - val_Precision: 0.6025 - val_Recall: 0.6025 - val_loss: 0.6641
Epoch 34/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.6307 - Recall: 0.6307 - loss: 0.6643 - val_Precision: 0.8199 - val_Recall: 0.8199 - val_loss: 0.6562
Epoch 35/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.6065 - Recall: 0.6065 - loss: 0.6703 - val_Precision: 0.5892 - val_Recall: 0.5892 - val_loss: 0.7195
Epoch 36/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.6289 - Recall: 0.6289 - loss: 0.6703 - val_Precision: 0.5533 - val_Recall: 0.5533 - val_loss: 0.6449
Epoch 37/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 21ms/step - Precision: 0.6222 - Recall: 0.6222 - loss: 0.6699 - val_Precision: 0.4949 - val_Recall: 0.4949 - val_loss: 0.7346
Epoch 38/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.6310 - Recall: 0.6310 - loss: 0.6647 - val_Precision: 0.6792 - val_Recall: 0.6792 - val_loss: 0.6517
Epoch 39/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.6419 - Recall: 0.6419 - loss: 0.6591 - val_Precision: 0.5666 - val_Recall: 0.5666 - val_loss: 0.6850
Epoch 40/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.6867 - Recall: 0.6867 - loss: 0.6032 - val_Precision: 0.7339 - val_Recall: 0.7339 - val_loss: 0.6271
Epoch 41/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.6889 - Recall: 0.6889 - loss: 0.6164 - val_Precision: 0.6176 - val_Recall: 0.6176 - val_loss: 0.6217
Epoch 42/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.6713 - Recall: 0.6713 - loss: 0.6331 - val_Precision: 0.8079 - val_Recall: 0.8079 - val_loss: 0.5764
Epoch 43/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.7178 - Recall: 0.7178 - loss: 0.5669 - val_Precision: 0.6562 - val_Recall: 0.6562 - val_loss: 0.5600
Epoch 44/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.7207 - Recall: 0.7207 - loss: 0.5426 - val_Precision: 0.6475 - val_Recall: 0.6475 - val_loss: 0.5862
Epoch 45/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.7274 - Recall: 0.7274 - loss: 0.5478 - val_Precision: 0.6006 - val_Recall: 0.6006 - val_loss: 0.7769
Epoch 46/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.7253 - Recall: 0.7253 - loss: 0.5418 - val_Precision: 0.5970 - val_Recall: 0.5970 - val_loss: 1.0727
Epoch 47/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.7130 - Recall: 0.7130 - loss: 0.5563 - val_Precision: 0.6227 - val_Recall: 0.6227 - val_loss: 0.6591
Epoch 48/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 21ms/step - Precision: 0.7422 - Recall: 0.7422 - loss: 0.4875 - val_Precision: 0.7771 - val_Recall: 0.7771 - val_loss: 0.4498
Epoch 49/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.7576 - Recall: 0.7576 - loss: 0.4829 - val_Precision: 0.8231 - val_Recall: 0.8231 - val_loss: 0.4132
Epoch 50/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.7092 - Recall: 0.7092 - loss: 0.5672 - val_Precision: 0.7233 - val_Recall: 0.7233 - val_loss: 0.5299
Epoch 51/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.7410 - Recall: 0.7410 - loss: 0.4977 - val_Precision: 0.8787 - val_Recall: 0.8787 - val_loss: 0.4300
Epoch 52/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - Precision: 0.7342 - Recall: 0.7342 - loss: 0.5029 - val_Precision: 0.8438 - val_Recall: 0.8438 - val_loss: 0.3808
Epoch 53/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.7409 - Recall: 0.7409 - loss: 0.4934 - val_Precision: 0.5653 - val_Recall: 0.5653 - val_loss: 0.8749
Epoch 54/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 6s 21ms/step - Precision: 0.7359 - Recall: 0.7359 - loss: 0.5273 - val_Precision: 0.8415 - val_Recall: 0.8415 - val_loss: 0.3838
Epoch 55/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.7690 - Recall: 0.7690 - loss: 0.4481 - val_Precision: 0.7762 - val_Recall: 0.7762 - val_loss: 0.4399
Epoch 56/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.7578 - Recall: 0.7578 - loss: 0.4690 - val_Precision: 0.6843 - val_Recall: 0.6843 - val_loss: 0.6670
Epoch 57/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.7511 - Recall: 0.7511 - loss: 0.4845 - val_Precision: 0.7146 - val_Recall: 0.7146 - val_loss: 0.5458
Epoch 58/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.7806 - Recall: 0.7806 - loss: 0.4358 - val_Precision: 0.8074 - val_Recall: 0.8074 - val_loss: 0.4081
Epoch 59/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.7745 - Recall: 0.7745 - loss: 0.4542 - val_Precision: 0.6641 - val_Recall: 0.6641 - val_loss: 0.7271
Epoch 60/100
272/272 ━━━━━━━━━━━━━━━━━━━━ 5s 20ms/step - Precision: 0.7512 - Recall: 0.7512 - loss: 0.4830 - val_Precision: 0.7578 - val_Recall: 0.7578 - val_loss: 0.4662

real    5m36.989s
user    5m48.208s
sys     2m19.473s
[1/1] Generating learning curve...
GPU is available

real    0m3.766s
user    0m4.794s
sys     0m1.648s
Generating training score
GPU is available
340/340 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step  

models/c.joblib: train: 

+-----+-----+
|5089 | 327 |
|1435 |4029 |
+-----+-----+

precision: 0.9249311294765841
recall: 0.737371888726208
f1: 0.8205702647657841


real    0m3.863s
user    0m4.594s
sys     0m0.911s
Generating validation score
GPU is available
85/85 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step 

models/c.joblib: train: 

+-----+-----+
|1323 |  59 |
| 337 |1001 |
+-----+-----+

precision: 0.9443396226415094
recall: 0.7481315396113603
f1: 0.8348623853211009


real    0m3.486s
user    0m4.137s
sys     0m0.883s
COMMENT

# This model is not good at all
# I am just going to stick with model b and AdaBoost