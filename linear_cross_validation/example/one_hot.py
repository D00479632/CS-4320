#!/usr/bin/env python3

import sklearn.pipeline
import sklearn.preprocessing
import sklearn.base

import pandas as pd

# I am a little confused, is this a pipeline or an element to a pipeline?

# This makes it possible to choose the data columns that we want to use or the 
# data that we think is most important.
class DataFrameSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    '''
    It inherits from sklearn so that it can behave like an element in a pipeline
    '''
    
    def __init__(self, do_predictors=True, do_numerical=True):
        # TODO: add all the other features
        # Initialize the selector with options for predictors and numerical features
        self.mCategoricalPredictors = ["RoofMatl"]  
        self.mNumericalPredictors = ["BedroomAbvGr"]  
        self.mLabels = ["SalePrice"]  
        # If do_numerical == True it keeps the columns that we specified were numerical
        # If do_numerical == False it keeps the columns that we specified were categorical
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
        # No fitting necessary for this transformer
        self.is_fitted_ = True  # Mark as fitted
        return self

    def transform( self, X, y=None ):
        # Only keep columns selected (either numerical or categorical)
        values = X[self.mAttributes]  # Extract the selected columns from the DataFrame
        return values

# Load the dataset from a CSV file
filename = "data/train.csv"
data = pd.read_csv(filename, index_col=0)

# Create a pipeline for numerical features
items = []
items.append(("numerical-features-only", DataFrameSelector(do_predictors=True, do_numerical=True)))
num_pipeline = sklearn.pipeline.Pipeline(items) # Create a pipeline for numerical features

# Create a pipeline for categorical features
items = []
items.append(("categorical-features-only", DataFrameSelector(do_predictors=True, do_numerical=False)))
# change categories to specific categories if you want to leave others out.
# handle_unknown is just in case there is a categorie that it hasn't seen
items.append(("encode-category-bits", sklearn.preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')))
cat_pipeline = sklearn.pipeline.Pipeline(items)

# Combine numerical and categorical pipelines into a single pipeline
items = []
items.append(("numerical", num_pipeline))
items.append(("categorical", cat_pipeline))
# This makes a pipeline out of pipelines
pipeline = sklearn.pipeline.FeatureUnion(transformer_list=items)


pipeline.fit(data) # Fit the pipeline to the training data
data_transform = pipeline.transform(data)  # Transform the data using the fitted pipeline
print(data_transform.shape)
# print(data_transform)
