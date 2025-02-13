#!/usr/bin/env python3

import sys
import argparse
import logging
import os.path

import pandas as pd
import numpy as np
import sklearn.ensemble
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.base
import sklearn.metrics
import sklearn.impute
import joblib

# This makes it possible to choose the data columns that we want to use or the 
# data that we think is most important.
class DataFrameSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    '''
    It inherits from sklearn so that it can behave like an element in a pipeline
    '''
    
    def __init__(self, do_predictors=True, do_numerical=True):
        # Initialize the selector with options for predictors and numerical features
        # TODO: OverallQual, OverallCond
        self.mCategoricalPredictors = ["MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", 
                                       "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", 
                                       "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation",
                                       "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",  "BsmtFinType2", "Heating", "HeatingQC", 
                                       "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish",
                                       "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "MoSold", "SaleType", 
                                       "SaleCondition"]  
        self.mNumericalPredictors = ["LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "MasVnrArea", 
                                     "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", 
                                     "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "TotRmsAbvGrd",
                                     "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", 
                                     "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "YrSold"]  
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

def get_test_filename(test_file, filename):
    ''' 
    Generates a test filename based on the provided filename. 
    If test_file is empty, it constructs a new filename by appending '-test.csv' to the base name.
    '''
    if test_file == "":
        basename = get_basename(filename)
        test_file = "data/{}-test.csv".format(basename)
    return test_file

def get_basename(filename):
    ''' 
    Extracts the base name from a given filename. 
    It removes the '-train' suffix if present and logs the components of the filename.
    '''
    root, ext = os.path.splitext(filename)
    dirname, basename = os.path.split(root)
    logging.info("root: {}  ext: {}  dirname: {}  basename: {}".format(root, ext, dirname, basename))

    stub = "-train"
    if basename[len(basename)-len(stub):] == stub:
        basename = basename[:len(basename)-len(stub)]

    return basename

def get_model_filename(model_file, filename):
    ''' 
    Generates a model filename based on the provided filename. 
    If model_file is empty, it constructs a new filename by appending '-model.joblib' to the base name.
    '''
    if model_file == "":
        basename = get_basename(filename)
        model_file = "models/{}-model.joblib".format(basename)
    return model_file

#def get_data(my_args, filename):
#   """
#   Assumes column 0 is the instance index stored in the
#   csv file.  If no such column exists, remove the
#   index_col=0 parameter.
#   """
#   data = pd.read_csv(filename, index_col=0)
#   data = data.dropna(subset=[my_args.label])
#   return data

def load_data(my_args, filename):
    ''' 
    Loads the data and separates it into features and labels. 
    It calls get_data to load the data and get_feature_and_label_names.
    '''
    data = pd.read_csv(filename, index_col=0)
    if my_args.label in data:
        data = data.dropna(subset=[my_args.label])

    feature_columns, label_column = get_feature_and_label_names(my_args, data)
    X = data[feature_columns]
    # When predicting we are going to have a problem here since we don't have a label
    # TODO: write some like if label_colum in data then do that else pass
    if label_column in data:
        y = data[label_column]
    else: 
        # This only happens on Test data
        y = None
    return X, y

def get_feature_and_label_names(my_args, data):
    ''' 
    Retrieves the feature and label names from the data based on user arguments. 
    If no features are specified, it defaults to all non-label columns.
    '''
    label_column = my_args.label
    feature_columns = my_args.features

    if label_column in data.columns:
        label = label_column
    else:
        label = ""

    features = []
    if feature_columns is not None:
        for feature_column in feature_columns:
            if feature_column in data.columns:
                features.append(feature_column)

    # no features specified, so add all non-labels
    if len(features) == 0:
        for feature_column in data.columns:
            if feature_column != label:
                features.append(feature_column)

    return features, label

def make_numerical_feature_pipeline(my_args):
    ''' 
    Creates a pipeline for processing numerical features. 
    It includes strategies for handling missing data, polynomial features, and scaling.
    '''
    items = []

    items.append(("numerical-features-only", DataFrameSelector(do_predictors=True, do_numerical=True)))
    if my_args.numerical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.numerical_missing_strategy)))
    if my_args.use_polynomial_features:
        items.append(("polynomial-features", sklearn.preprocessing.PolynomialFeatures(degree=my_args.use_polynomial_features)))
    if my_args.use_scaler:
        # If with_means=True, center the data before scaling. This does not work 
        # (and will raise an exception) when attempted on sparse matrices, because centering them entails building a dense matrix
        # which in common use cases is likely to be too large to fit in memory.
        items.append(("scaler", sklearn.preprocessing.StandardScaler(with_mean=False)))
    items.append(("noop", PipelineNoop()))
    
    numerical_pipeline = sklearn.pipeline.Pipeline(items)
    return numerical_pipeline

def make_categorical_feature_pipeline(my_args):
    items = []
    items.append(("categorical-features-only", DataFrameSelector(do_predictors=True, do_numerical=False)))
    items.append(("encode-category-bits", sklearn.preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')))
    if my_args.use_polynomial_features:
        items.append(("polynomial-features", sklearn.preprocessing.PolynomialFeatures(degree=my_args.use_polynomial_features)))
    if my_args.use_scaler:
        # If with_means=True, center the data before scaling. This does not work 
        # (and will raise an exception) when attempted on sparse matrices, because centering them entails building a dense matrix
        # which in common use cases is likely to be too large to fit in memory.
        items.append(("scaler", sklearn.preprocessing.StandardScaler(with_mean=False)))
    # We use this just in case all myargs if statements are false the pipeline will have something (just a dummy)
    items.append(("noop", PipelineNoop()))

    categorical_pipeline = sklearn.pipeline.Pipeline(items)
    return categorical_pipeline

def make_features(my_args):
    items = []
    items.append(("numerical-features", make_numerical_feature_pipeline(my_args)))
    items.append(("categorical-features", make_categorical_feature_pipeline(my_args)))

    pipeline = sklearn.pipeline.FeatureUnion(transformer_list=items)
    return pipeline

def make_SGD_fit_pipeline(my_args):
    ''' 
    Creates a pipeline for fitting a Stochastic Gradient Descent (SGD) model. 
    It includes the numerical feature pipeline and the SGD regressor.
    '''
    items = []
    #items.append(("features", make_numerical_feature_pipeline(my_args)))
    items.append(("features", make_features(my_args)))

    if my_args.model_type == "SGD":
        # models to try: sklearn.linear_model.SGDRegressor(), sklearn.linear_model.Ridge(), sklearn.linear_model.RidgeCV(), 
        '''
        sklearn.linear_model.RidgeCV(
            alphas=np.logspace(-3, 3, 50),  # Wider range of alpha values
            cv=5,  # Use 5-fold CV instead of LOO
            fit_intercept=True,  # Keep it unless data is already centered
            gcv_mode='svd'  # More stable for high-dimensional data
        )))
        '''
        items.append(("model",sklearn.linear_model.Lasso(max_iter=100000)))
    elif my_args.model_type == "linear":
        # TODO: find good linnear classifier
        raise Exception("Need to put a good linear classifier here.")
    elif my_args.model_type == "SVM":
        items.append(("model", sklearn.svm.SVC()))
    elif my_args.model_type == "boost":
        items.append(("model", sklearn.ensemble.GradientBoostingClassifier()))
    elif my_args.model_type == "forest":
        items.append(("model", sklearn.ensemble.RandomForestClassifier()))
    elif my_args.model_type == "tree":
        items.append(("model", sklearn.tree.DecisionTreeClassifier()))
    return sklearn.pipeline.Pipeline(items)

def do_fit(my_args):
    ''' 
    Fits the model using the training data. 
    It loads the data, creates the pipeline, and saves the fitted model to a file.
    '''
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_SGD_fit_pipeline(my_args)
    pipeline.fit(X, y)

    model_file = get_model_filename(my_args.model_file, train_file)

    joblib.dump(pipeline, model_file)
    return

def show_score(my_args):
    ''' 
    Displays the training and testing scores of the model. 
    It calculates the scores using the regressor and prints them based on user preference.
    '''
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    test_file = get_test_filename(my_args.test_file, train_file)
    if not os.path.exists(test_file):
        raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    X_train, y_train = load_data(my_args, train_file)
    # y_test will be None here 
    X_test, y_test = load_data(my_args, test_file)
    pipeline = joblib.load(model_file)
    regressor = pipeline['model']
    
    basename = get_basename(train_file)
    score_train = regressor.score(pipeline['features'].transform(X_train), y_train)
    # We cannot do show test because y_test is None
    if my_args.show_test:
        print("YOU CANNOT DO SHOW TEST BECAUSE WE DONT HAVE THE LABEL FOR TEST DATA")
        #score_test = regressor.score(pipeline['features'].transform(X_test), y_test)
        #print("{}: train_score: {} test_score: {}".format(basename, score_train, score_test))
    else:
        print("{}: train_score: {}".format(basename, score_train))
    return

def show_loss(my_args):
    ''' 
    Displays the training and testing loss of the model. 
    It calculates the loss using mean squared error, mean absolute error, and R2 score.
    '''
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    test_file = get_test_filename(my_args.test_file, train_file)
    if not os.path.exists(test_file):
        raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    X_train, y_train = load_data(my_args, train_file)
    X_test, y_test = load_data(my_args, test_file)
    pipeline = joblib.load(model_file)

    y_train_predicted = pipeline.predict(X_train)
    y_test_predicted = pipeline.predict(X_test)

    basename = get_basename(train_file)
    
    loss_train = sklearn.metrics.mean_squared_error(y_train, y_train_predicted)
    if my_args.show_test:
        loss_test = sklearn.metrics.mean_squared_error(y_test, y_test_predicted)
        print("{}: L2(MSE) train_loss: {} test_loss: {}".format(basename, loss_train, loss_test))
    else:
        print("{}: L2(MSE) train_loss: {}".format(basename, loss_train))

    loss_train = sklearn.metrics.mean_absolute_error(y_train, y_train_predicted)
    if my_args.show_test:
        loss_test = sklearn.metrics.mean_absolute_error(y_test, y_test_predicted)
        print("{}: L1(MAE) train_loss: {} test_loss: {}".format(basename, loss_train, loss_test))
    else:
        print("{}: L1(MAE) train_loss: {}".format(basename, loss_train))

    loss_train = sklearn.metrics.r2_score(y_train, y_train_predicted)
    if my_args.show_test:
        loss_test = sklearn.metrics.r2_score(y_test, y_test_predicted)
        print("{}: R2 train_loss: {} test_loss: {}".format(basename, loss_train, loss_test))
    else:
        print("{}: R2 train_loss: {}".format(basename, loss_train))
    return

def show_model(my_args):
    ''' 
    Displays information about the fitted model, including coefficients, intercept, and scaler details. 
    It retrieves and prints the model's parameters.
    '''
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    test_file = get_test_filename(my_args.test_file, train_file)
    if not os.path.exists(test_file):
        raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    pipeline = joblib.load(model_file)
    regressor = pipeline['model']
    features = pipeline['features']

    print("Model Information:")
    print("coef_: {}".format(regressor.coef_))
    print("intercept_: {}".format(regressor.intercept_))
    print("n_iter_: {}".format(regressor.n_iter_))
    print("n_features_in_: {}".format(regressor.n_features_in_))


    try:
        scaler = features["scaler"]
        print("scaler.mean_: {}".format(scaler.mean_))
        print("scaler.var_: {}".format(scaler.var_))
    except:
        print("No scaler.")
    return

# Function meant to just use on the pipeline 
def do_predict(my_args):

    test_file = my_args.test_file
    if not os.path.exists(test_file):
        raise Exception("testing data file: {} does not exist.".format(test_file))
    
    model_file = get_model_filename(my_args.model_file, test_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    # y_test is None here
    X_test, y_test = load_data(my_args, test_file)
    pipeline = joblib.load(model_file)

    y_test_predicted = pipeline.predict(X_test)

    merged = X_test.index.to_frame()
    merged['SalePrice'] = y_test_predicted
    merged.to_csv("predictions/predictions.csv", index=False)

    return

def do_cross(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_SGD_fit_pipeline(my_args)

    cv_results = sklearn.model_selection.cross_validate(pipeline, X, y, cv=3, n_jobs=-1, verbose=3, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'),)

    print("R2:", cv_results['test_r2'], cv_results['test_r2'].mean())
    print("MSE:", cv_results['test_neg_mean_squared_error'], cv_results['test_neg_mean_squared_error'].mean())
    print("MAE:", cv_results['test_neg_mean_absolute_error'], cv_results['test_neg_mean_absolute_error'].mean())

    return

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Fit Data With Linear Regression Using Pipeline')
    parser.add_argument('action', default='train',
                        choices=[ "train", "score", "loss", "show-model", "predict", "cross"], 
                        nargs='?', help="desired action")
    # data/train.csv
    parser.add_argument('--train-file',    '-t', default="",    type=str,   help="name of file with training data")
    parser.add_argument('--test-file',     '-T', default="",    type=str,   help="name of file with test data (default is constructed from train file name)")
    parser.add_argument('--model-file',    '-m', default="",    type=str,   help="name of file for the model (default is constructed from train file name when fitting)")
    # Curtis added this one after.
    parser.add_argument('--model-type',    '-M', default="SGD", type=str,   choices=["SGD", "linear", "SVG", "boost", "forest", "tree"], help="Model type")
    parser.add_argument('--random-seed',   '-R', default=2468,type=int,help="random number seed (-1 to use OS entropy)")
    parser.add_argument('--features',      '-f', default=None, action="extend", nargs="+", type=str,
                        help="column names for features")
    parser.add_argument('--label',         '-l', default="SalePrice",   type=str,   help="column name for label")
    parser.add_argument('--use-polynomial-features', '-p', default=0,         type=int,   help="degree of polynomial features.  0 = don't use (default=0)")
    parser.add_argument('--use-scaler',    '-s', default=0,         type=int,   help="0 = don't use scaler, 1 = do use scaler (default=0)")
    parser.add_argument('--numerical-missing-strategy', default="",   type=str,   help="strategy for missing numerical information")
    parser.add_argument('--show-test',     '-S', default=0,         type=int,   help="0 = don't show test loss, 1 = do show test loss (default=0)")

    my_args = parser.parse_args(argv[1:])

    allowed_numerical_missing_strategies = ("mean", "median", "most_frequent")
    if my_args.numerical_missing_strategy != "":
        if my_args.numerical_missing_strategy not in allowed_numerical_missing_strategies:
            raise Exception("Missing numerical strategy {} is not in the allowed list {}.".format(my_args.numerical_missing_strategy, allowed_numerical_missing_strategies))

    
    return my_args

def main(argv):
    my_args = parse_args(argv)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)

    if my_args.action == 'train':
        do_fit(my_args)
    elif my_args.action == "score":
        show_score(my_args)
    elif my_args.action == "loss":
        show_loss(my_args)
    elif my_args.action == "show-model":
        show_model(my_args)
    elif my_args.action == "predict":
        do_predict(my_args)
    elif my_args.action == "cross":
        do_cross(my_args)
    else:
        raise Exception("Action: {} is not known.".format(my_args.action))
        
    return

if __name__ == "__main__":
    main(sys.argv)
    
