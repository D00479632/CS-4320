# LINNEAR REGRESSION 

## BIG PICTURE

Following the flow chart from the powerpoint I first went into [kaggle](https://www.kaggle.com/datasets/stealthtechnologies/predict-student-performance-dataset) and 
I read about the dataset, the description and the key features. They even offer some histograms with the data. 

## GET DATA

Once I felt like I understood the data enough I downloaded it and placed it in my `cvs` directory where I plan to have all the csv files so that 
everything is a little bit more organized. 

## EXPLORE/VISUALIZE

I am now working on getting the histograms and the scatter plots. 

Looking at the histograms its crazy how many students got less than 40% on
their grade. I also see a peak of students in between 2.5 and 5 hours of studying. The other plots are very distributed.

Looking at the scatters I can already tell that the Stochastic Gradient Decent (SGD) model is not going to work very well because the data doesn't look 
linear, if anything, the study hours and socioeconomic score remind me a little of the absolute value graph. However, sleep hours and attendance just look 
crazy to me.

## PREPARE DATA

Before starting with the data manipulation I am going to make sure to split the data and set aside my precious `testing_data`

### FIRST TRY

On my first try I am not going to do anything. This is like a `random agent`, I want to see how the model learns and what the loss is without normalizing or 
doing anything else. 

Before going any further and training the first model I want to check that the split data is not corrupt so I will quickly look through it to see if there 
is anything weird. Everything looks good and the data split was a 80/20.

### SECOND TRY

Now I will normalize the data so that everything has the same weight and everything will be in a range 0-1. I will train the model again and see what the results are.

### THIRD TRY

I will give one last chance to SGDRegressor and try and pass in some parameters to make the training better and see if we see any change in the results.
For what I have been reading in 
[SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor)
I can do a lot of different stuff but I will only try and modify `loss='epsilon_insensitive', penalty='l1'` I also found there is a different option 
for the scaler called [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) and it uses the median and 
the interquartile range (IQR), instead of the mean and standard deviation. Maybe I give it a try sometime.

### FOURTH TRY

Let's change up the model, looking at the linear model that scikit offers 
[Bayesian Regression](https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression) caught my attention because the advantages listed say: 
It adapts to the data at hand and It can be used to include regularization parameters in the estimation procedure.

For the scaler I will also try something different because the function could not be linear and I will use the `sklearn.preprocessing.PolynomialFeatures` 
that we saw in class to change the degree.

## SELECT/TRAIN MODEL

### FIRST TRY

`R^2: -2.5225381231481267e+18`

`MSE: 2.1732074172217885e+20`

`MAE: 13605023711.61613`

`-8371645057.788 + (7896722920.168*Socioeconomic Score) + (-2659687482.275*Study Hours) + (-2296621706.469*Sleep Hours) + (358374457.352*Attendance (%))`

As we can see here this is very bad results. The R^2 error is huge and looking at the formula it says negative sleep hours and 
study hours which is weird and probably very wrong.

### SECOND TRY

By only adding a `sklearn.preprocessing.StandardScaler()` we can see better results:

`R^2: 0.7655152376115248`

`MSE: 20.201241764075387`

`MAE: 3.445794172619854`

`17.440 + (12.073*Socioeconomic Score) + ( 4.238*Study Hours) + ( 0.169*Sleep Hours) + (-0.070*Attendance (%))`

I can tell from this that what matters the most is Socioeconomic Score and Study Hours so maybe giving some extra parameters to the SGDRegressor function we can get it to be even better.

### THIRD TRY

`R^2: 0.7395243659356261`

`MSE: 22.44039742193444`

`MAE: 3.3411905902931753`

`19.103 + ( 8.163*Socioeconomic Score) + ( 4.123*Study Hours) + ( 0.096*Sleep Hours) + (-0.062*Attendance (%))`

Comparing it to the first try we did way better but comparing it to the second try we actually did a little worse so I will just keep what I had on my second
model for my best model.

### FOURTH TRY

`R^2: 0.9259630778286498`

`MSE: 6.378400664575503`

`MAE: 1.9606127086337626`

The formula didn't work since I used different scaler and regressor and the attributes were not the same. However, we see way more improvement in R^2. This
is my best so far and I will stick with it.

## FINE-TUNE MODEL

(This is all the tries above)