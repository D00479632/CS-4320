#!/usr/bin/env python3


################################################################
#
# These custom classes help with pipeline building and debugging
#

import sklearn.base
import pandas as pd

class PipelineNoop(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Just a placeholder with no actions on the data.
    """
    
    def __init__(self):
        return

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        return X

class Printer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Pipeline member to display the data at this stage of the transformation.
    """
    
    def __init__(self, title):
        self.title = title
        return

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        print("{}::type(X)".format(self.title), type(X))
        print("{}::X.shape".format(self.title), X.shape)
        if not isinstance(X, pd.DataFrame):
            print("{}::X[0]".format(self.title), X[0])
        print("{}::X".format(self.title), X)
        return X

class DataFrameSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    '''
    Here are the features for train.csv:
    Name,Gender,Age,City,Working Professional or Student,Profession,Academic Pressure,Work Pressure,
    CGPA,Study Satisfaction,Job Satisfaction,Sleep Duration,Dietary Habits,Degree,
    Have you ever had suicidal thoughts ?,Work/Study Hours,Financial Stress,Family History of Mental Illness,Depression
    '''
    
    def __init__(self, do_predictors=True, do_numerical=True):
        # I am not going to include clinical_notes because I don't think it's good for training the model since we would get so many features after one hot encode.
        self.mCategoricalPredictors = ["gender", "location", "smoking_history"]
        self.mNumericalPredictors = ["year", "age", "race:AfricanAmerican", "race:Asian", "race:Caucasian", "race:Hispanic", "race:Other", "hypertension", 
                                     "heart_disease", "bmi", "hbA1c_level", "blood_glucose_level"]
        self.mLabels = ['diabetes']
        self.do_numerical = do_numerical
        self.do_predictors = do_predictors
        
        if do_predictors:
            if do_numerical:
                self.mAttributes = self.mNumericalPredictors
            else:
                self.mAttributes = self.mCategoricalPredictors                
        else:
            self.mAttributes = self.mLabels
            
        return

    def fit( self, X, y=None ):
        # no fit necessary
        self.is_fitted_ = True
        return self

    def transform( self, X, y=None ):
        # only keep columns selected
        values = X[self.mAttributes]
        return values

#
# These custom classes help with pipeline building and debugging
#
################################################################
