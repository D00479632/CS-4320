#!/bin/bash

# Here is a little more information about the labels
# id: Unique identifier for each record or entry in the dataset
# person_age: The age of the person applying for the loan
# person_income: The annual income of the person applying for the loan
# person_home_ownership: The type of home ownership the person has (e.g., 'own', 'rent', 'mortgage')
# person_emp_length: The number of years the person has been employed
# loan_intent: The purpose or intent of the loan (e.g., 'education', 'medical', 'personal')
# loan_grade: A grade assigned to the loan based on the risk level (e.g., 'A', 'B', 'C')
# loan_amnt: The total amount of money being requested for the loan
# loan_int_rate: The interest rate applied to the loan, typically as an annual percentage rate (APR)
# loan_percent_income: The percentage of the person's income that is allocated to loan repayment
# cb_person_default_on_file: A binary indicator (yes/no) showing if the person has a history of defaulting on loans
# cb_person_cred_hist_length: The length of the person's credit history in years
# loan_status: The current status of the loan 0 or 1

# Here I will store all the comands for easier use and faster report afterwards

# First lets start with looking at the data and generating the histograms and scatter plots
#./display_data.py all --data-file data/train.csv
# Because everything is so polarized the scatter plot function doesn't work well for this data 
# and you can barely see at the top and bottom all the points. So for now I will just take a look
# at the histograms

# For what I know right now I think that RandomForestClassifier or GradientBoostingClassifier will do the best out of
# the different models we have set up. But like always I want to start with the SGDClassifier to get used to the process.

#./pipeline.py random-search --model-type SGD --train-file data/train.csv --model-file models/SGDClassifier.joblib --search-grid-file models/SearchGridSGDClassifier.joblib --use-polynomial-features 2 --use-scaler 1 --categorical-missing-strategy most_frequent --numerical-missing-strategy median --n-search-iterations 10 

# Let's see what our best parameters are
#./pipeline.py show-best-params --model-type SGD --train-file data/train.csv --search-grid-file models/SearchGridSGDClassifier.joblib 

#Best Score: 0.9144001833459651
#Best Params:
#{   'features__categorical__categorical-features-only__do_numerical': False,
#    'features__categorical__categorical-features-only__do_predictors': True,
#    'features__categorical__encode-category-bits__categories': 'auto',
#    'features__categorical__encode-category-bits__handle_unknown': 'ignore',
#    'features__categorical__missing-data__strategy': 'most_frequent',
#    'features__numerical__missing-data__strategy': 'median',
#    'features__numerical__numerical-features-only__do_numerical': True,
#    'features__numerical__numerical-features-only__do_predictors': True,
#    'features__numerical__polynomial-features__degree': 2,
#    'model__alpha': 0.0001,
#    'model__average': False,
#    'model__class_weight': None,
#    'model__early_stopping': False,
#    'model__epsilon': 0.1,
#    'model__eta0': 0.0,
#    'model__fit_intercept': True,
#    'model__l1_ratio': 0.15,
#    'model__learning_rate': 'optimal',
#    'model__loss': 'log_loss',
#    'model__max_iter': 1000,
#    'model__n_iter_no_change': 5,
#    'model__n_jobs': -1,
#    'model__penalty': None,
#    'model__power_t': 0.5,
#    'model__random_state': None,
#    'model__shuffle': True,
#    'model__tol': 0.001,
#    'model__validation_fraction': 0.1,
#    'model__verbose': 0,
#    'model__warm_start': False}

# Maybe we overfitted our model so lets look at the confusion matrix (make sure to change the make_fit_pipeline_classification with parameters found)
#./pipeline.py confusion-matrix --model-type SGD --train-file data/train.csv --search-grid-file models/SearchGridSGDClassifier.joblib 

#     t/p      F     T 
#        F 48509.0 1786.0 
#        T 6518.0 1832.0 
#Precision: 0.506
#Recall:    0.219
#F1:        0.306
# This is very poor 50.6% of the positive predictions were actually correct so we have a 50/50 model. Only 21.9% of the actual positives were correctly identified.
# But if I change the make_fit_pipeline_classification and delete the best parameters we found then we get this confusion matrix:
#     t/p      F     T 
#        F 42871.0 7424.0 
#        T 3595.0 4755.0 
#Precision: 0.390
#Recall:    0.569
#F1:        0.463
# And I can tell that the hyperparameters that I found make the model way better than with the default by a lot (30% to 50%)
# Still, I am not going to do proba on this model because I know is not great

# Now with RandomForestClassifier
#./pipeline.py random-search --model-type forest --train-file data/train.csv --model-file models/RandomForestClassifier.joblib --search-grid-file models/SearchGridRandomForestClassifier.joblib --use-polynomial-features 2 --use-scaler 1 --categorical-missing-strategy most_frequent --numerical-missing-strategy median --n-search-iterations 10 
# This took over 45 minutes I really hope it was not a waste of time

# Let's see the best found parameters
#./pipeline.py show-best-params --model-type forest --train-file data/train.csv --search-grid-file models/SearchGridRandomForestClassifier.joblib 
#Best Score: 0.94988493408149
#Best Params:
#{   'features__categorical__categorical-features-only__do_numerical': False,
#    'features__categorical__categorical-features-only__do_predictors': True,
#    'features__categorical__encode-category-bits__categories': 'auto',
#    'features__categorical__encode-category-bits__handle_unknown': 'ignore',
#    'features__categorical__missing-data__strategy': 'most_frequent',
#    'features__numerical__missing-data__strategy': 'median',
#    'features__numerical__numerical-features-only__do_numerical': True,
#    'features__numerical__numerical-features-only__do_predictors': True,
#    'features__numerical__polynomial-features__degree': 2,
#    'model__bootstrap': True,
#    'model__ccp_alpha': 0.0,
#    'model__class_weight': None,
#    'model__criterion': 'gini',
#    'model__max_depth': None,
#    'model__max_features': None,
#    'model__max_leaf_nodes': None,
#    'model__max_samples': None,
#    'model__min_impurity_decrease': 0.0,
#    'model__min_samples_leaf': 1,
#    'model__min_samples_split': 2,
#    'model__min_weight_fraction_leaf': 0.0,
#    'model__monotonic_cst': None,
#    'model__n_estimators': 500,
#    'model__n_jobs': -1,
#    'model__oob_score': False,
#    'model__random_state': None,
#    'model__verbose': 0,
#    'model__warm_start': False}

# Lets look at the confusion matrix (make sure to change the make_fit_pipeline_classification with parameters found)
#./pipeline.py confusion-matrix --model-type forest --train-file data/train.csv --search-grid-file models/SearchGridRandomForestClassifier.joblib
#     t/p      F     T 
#        F 49754.0 541.0 
#        T 2355.0 5995.0 
#Precision: 0.917
#Recall:    0.718
#F1:        0.805

# Comparing it to the last one we did way better with 90% of the the positive predictions were actually correct. 70% of the true positives were identified.
# I will do proba to submit to kaggle and see how I'm doing so far. I will still try more models afterwards.

# Doing a precision-recall-plot
#./pipeline.py precision-recall-plot --model-type forest --train-file data/train.csv --image-file plots/precisionRecallPlotRandomForest.png 

# Now a pr-curve
#./pipeline.py pr-curve --model-type forest --train-file data/train.csv --image-file plots/prCurveRandomForest.png 

#./pipeline.py proba --model-type forest --test-file data/test.csv --model-file models/RandomForestClassifier.joblib --proba-file predictions/predictionsRandomForestProba.csv
# After submitted to kaggle I got a 0.93810 which is not bad for my first try. I went over what I needed for the assignment

# Moving on to the next I want to try the GradientBoostingClassifier
#./pipeline.py random-search --model-type boost --train-file data/train.csv --model-file models/GradientBoostingClassifier.joblib --search-grid-file models/SearchGridGradientBoostingClassifier.joblib --use-polynomial-features 2 --use-scaler 1 --categorical-missing-strategy most_frequent --numerical-missing-strategy median --n-search-iterations 10 
# It has been running for an hour and half now, I am going to go to bed and hope it is done
# It took almost three hours.

# Lets see the best parameters that were found
#./pipeline.py show-best-params --model-type boost --train-file data/train.csv --search-grid-file models/SearchGridGradientBoostingClassifier.joblib 
#Best Score: 0.9489641298743177
#Best Params:
#{   'features__categorical__categorical-features-only__do_numerical': False,
#    'features__categorical__categorical-features-only__do_predictors': True,
#    'features__categorical__encode-category-bits__categories': 'auto',
#    'features__categorical__encode-category-bits__handle_unknown': 'ignore',
#    'features__categorical__missing-data__strategy': 'most_frequent',
#    'features__numerical__missing-data__strategy': 'median',
#    'features__numerical__numerical-features-only__do_numerical': True,
#    'features__numerical__numerical-features-only__do_predictors': True,
#    'features__numerical__polynomial-features__degree': 2,
#    'model__ccp_alpha': 0.0,
#    'model__criterion': 'squared_error',
#    'model__init': None,
#    'model__learning_rate': 0.1,
#    'model__loss': 'exponential',
#    'model__max_depth': None,
#    'model__max_features': None,
#    'model__max_leaf_nodes': None,
#    'model__min_impurity_decrease': 0.0,
#    'model__min_samples_leaf': 1,
#    'model__min_samples_split': 2,
#    'model__min_weight_fraction_leaf': 0.0,
#    'model__n_estimators': 200,
#    'model__n_iter_no_change': None,
#    'model__random_state': None,
#    'model__subsample': 0.9,
#    'model__tol': 0.0001,
#    'model__validation_fraction': 0.1,
#    'model__verbose': 0,
#    'model__warm_start': False}

# Now lets see how good the confusion matrix is
#./pipeline.py confusion-matrix --model-type boost --train-file data/train.csv --search-grid-file models/SearchGridGradientBoostingClassifier.joblib
#     t/p      F     T 
#        F 49775.0 520.0 
#        T 2359.0 5991.0 
#Precision: 0.920
#Recall:    0.717
#F1:        0.806

# Comparing it to the RandomForest it barely improved. However, I still want to submit it to see what the difference would be.

# Doing a precision-recall-plot
#./pipeline.py precision-recall-plot --model-type boost --train-file data/train.csv --image-file plots/precisionRecallPlotGradientBoosting.png 

# Now a pr-curve
#./pipeline.py pr-curve --model-type forest --train-file data/train.csv --image-file plots/prCurveGradientBoosting.png 

#./pipeline.py proba --model-type boost --test-file data/test.csv --model-file models/GradientBoostingClassifier.joblib --proba-file predictions/predictionsGradientBoostingProba.csv
# After submitted to kaggle I got a 0.94164 which is not much better than the random forest was.