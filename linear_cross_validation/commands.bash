#!/bin/bash

# Cross validate with sklearn.linear_model.SGDRegressor() in the pipeline
#R2: [-3.24233543e+20 -9.38801652e+18 -4.10333630e+20] -2.4798506289209393e+20
#MSE: [-1.97532280e+30 -6.36545824e+28 -2.47666131e+30] -1.505212896143166e+30
#MAE: [-1.40545116e+15 -2.52229589e+14 -1.57373161e+15] -1077137452086678.6
#./pipeline.py cross --train-file data/train.csv --use-polynomial-features 0  --use-scaler 1 --numerical-missing-strategy median  

# Cross validate with sklearn.linear_model.Ridge()
#R2: [0.87770503 0.82918791 0.74014908] 0.8156806736003103
#MSE: [-7.45055658e+08 -1.15817566e+09 -1.56838895e+09] -1157206753.1387541
#MAE: [-18573.33349128 -18684.01281199 -18810.67298724] -18689.339763503896
#./pipeline.py cross --train-file data/train.csv --use-polynomial-features 0  --use-scaler 1 --numerical-missing-strategy median  
# This one did way better but still got MAE -18689.3 which is off by $-18689.3 so not good yet


# Cross validate with sklearn.linear_model.RidgeCV()
#R2: [0.87772316 0.83755597 0.74479315] 0.820024096734628
#MSE: [-7.44945211e+08 -1.10143677e+09 -1.54035857e+09] -1128913517.5318768
#MAE: [-18617.85723931 -18114.11123078 -18592.81938117] -18441.595950418905
#./pipeline.py cross --train-file data/train.csv --use-polynomial-features 0  --use-scaler 1 --numerical-missing-strategy median  
# Very close to Ridge

# For now I will just do RidgeCV() and Ridge()
#./pipeline.py train --train-file data/train.csv --model-file models/ridgeCV-model.joblib --use-polynomial-features 0  --use-scaler 1 --numerical-missing-strategy median  
#./pipeline.py train --train-file data/train.csv --model-file models/ridge-model.joblib --use-polynomial-features 0  --use-scaler 1 --numerical-missing-strategy median  

# Lets look at the loss with the data it was trained on (for RidgeCV)
#./pipeline.py loss --train-file data/train.csv --test-file data/train.csv --model-file models/ridgeCV-model.joblib 
#train: L2(MSE) train_loss: 418293859.47137684
#train: L1(MAE) train_loss: 13198.388821823875
#train: R2 train_loss: 0.9336756173430205
# We can see it does really good which could be bad and it could mean that we are overfitting

# Lets look at the loss with the data it was trained on (for Ridge)
#./pipeline.py loss --train-file data/train.csv --test-file data/train.csv --model-file models/ridge-model.joblib 
#train: L2(MSE) train_loss: 418293859.47137684
#train: L1(MAE) train_loss: 13198.388821823875
#train: R2 train_loss: 0.9336756173430205
# They are identical. Still probably overfitting

# Lets fine tune RidgeCV(
#    alphas=np.logspace(-3, 3, 50),  # Wider range of alpha values
#    cv=5,  # Use 5-fold CV instead of LOO
#    fit_intercept=True,  # Keep it unless data is already centered
#    gcv_mode='svd'  # More stable for high-dimensional data
#)
# Lets see what we should expect
#./pipeline.py cross --train-file data/train.csv --model-file models/ridgeCV-fine-model.joblib --use-polynomial-features 0  --use-scaler 1 --numerical-missing-strategy median  
#R2: [0.89234769 0.84808859 0.77796931] 0.8394685271761131
#MSE: [-6.55848457e+08 -1.03002135e+09 -1.34011641e+09] -1008662070.6822224
#MAE: [-16959.90322372 -17686.23526928 -17561.04486846] -17402.394453820645

# Now lets train the model
#./pipeline.py train --train-file data/train.csv --model-file models/ridgeCV-fine-model.joblib --use-polynomial-features 0  --use-scaler 1 --numerical-missing-strategy median  

# Lets look at the loss with the data it was trained on (for RidgeCV fine)
#./pipeline.py loss --train-file data/train.csv --test-file data/train.csv --model-file models/ridgeCV-fine-model.joblib 
#train: L2(MSE) train_loss: 535765977.97651047
#train: L1(MAE) train_loss: 14306.47414882316
#train: R2 train_loss: 0.9150493201530336
# This is a little worse but maybe it is less overfit

# Lets fine tune RidgeCV(
#    alphas=np.logspace(-3, 3, 50),  # Wider range of alpha values
#    cv=5,  # Use 5-fold CV instead of LOO
#    fit_intercept=True,  # Keep it unless data is already centered
#    gcv_mode='svd'  # More stable for high-dimensional data
#)
# and also add polynomial-features 2
# Lets see what we should expect
#./pipeline.py cross --train-file data/train.csv --model-file models/ridgeCV-fine-model.joblib --use-polynomial-features 2  --use-scaler 1 --numerical-missing-strategy median  
# [CV] END  neg_mean_absolute_error: (test=-17812.740) neg_mean_squared_error: (test=-1718295760.207) r2: (test=0.715) total time= 3.0min
# I stopped it after that because I saw it wasn't worth it to wait for the result since i was probably going to get similar scores

# My best model up to know are the RidgeCV, Ridge or RidgeCVfine, i will predict with RidgeCV and submit that to kaggle to see how I did
# Use this for submitting to kaggle:
#./pipeline.py predict --train-file data/train.csv --test-file data/test.csv --model-file models/ridgeCV-fine-model.joblib
# Score: 0.14651

# Doing some research on kaggle I found they use sklearn.linear_model.Lasso a lot so I will give it a try 
# Cross validate with sklearn.linear_model.Lasso(max_iterations=100000)
#R2: [0.87707974 0.82939818 0.73294526] 0.8131410606249699
#MSE: [-7.48865096e+08 -1.15674996e+09 -1.61186921e+09] -1172494755.370797
#MAE: [-18640.22920593 -18568.64260392 -18811.67510177] -18673.515637206685
#./pipeline.py cross --train-file data/train.csv --use-polynomial-features 0  --use-scaler 1 --numerical-missing-strategy median  
# This looks like it could be no so overfitted

# Lets train it.
#./pipeline.py train --train-file data/train.csv --model-file models/lasso-model.joblib --use-polynomial-features 0  --use-scaler 1 --numerical-missing-strategy median  
# Objective did not converge. You might want to increase the number of iterations.
# I will increase the iterations and run it again
# Now it worked

# Lets look at the loss with the data it was trained on (for lasso)
#./pipeline.py loss --train-file data/train.csv --test-file data/train.csv --model-file models/lasso-model.joblib 
#train: L2(MSE) train_loss: 416319831.39331555
#train: L1(MAE) train_loss: 13219.494667704925
#train: R2 train_loss: 0.9339886178584726
# This model to me seems overfitted so I'll stay with the other submission

# Now i want to try random forest
#./pipeline.py cross --train-file data/train.csv --model-type forest --use-polynomial-features 0  --use-scaler 1 --numerical-missing-strategy median  
#R2: [0.716761   0.72253017 0.71741345] 0.718901543182902
#MSE: [-1.71257442e+09 -1.48393991e+09 -2.11816946e+09] -1771561263.8532784
#MAE: [-26827.52977413 -25273.50308008 -27675.88271605] -26592.305190086277
# Doesn't look better than the other one but maybe it will make it not overfitted

# Im curious so I will train it 
#./pipeline.py train --train-file data/train.csv --model-type forest --model-file models/forest-model.joblib --use-polynomial-features 0  --use-scaler 1 --numerical-missing-strategy median  

# Lets look at the loss with the data it was trained on (for forest)
#./pipeline.py loss --train-file data/train.csv --test-file data/train.csv --model-file models/forest-model.joblib 
#train: L2(MSE) train_loss: 0.0
#train: L1(MAE) train_loss: 0.0
#train: R2 train_loss: 1.0
# Nevermind this is very overfitted jajajaja i want to still try and predict because I'm curious

# Getting the prediction
#./pipeline.py predict --train-file data/train.csv --test-file data/test.csv --model-file models/forest-model.joblib
# Sore: 0.21058

# The next steps I should have done is probably take some of the features out that I dont think are relevant