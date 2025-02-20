#!/usr/bin/env python3

################################################################
#
# These custom functions help with constructing common pipelines.
# They make use of my_args, and object that has been configured
# by the argparse module to match user requests.
#
from pipeline_elements import *
import sklearn.impute
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.tree

def make_numerical_predictor_params(my_args):
    params = { 
        "features__numerical__numerical-features-only__do_predictors" : [ True ],
        "features__numerical__numerical-features-only__do_numerical" : [ True ],
    }
    if my_args.numerical_missing_strategy:
        params["features__numerical__missing-data__strategy"] = [ 'median' ] # 'mean', 'most_frequent'
    if my_args.use_polynomial_features:
        params["features__numerical__polynomial-features__degree"] = [ 2 ] # [ 1, 2, 3 ]

    return params

def make_categorical_predictor_params(my_args):
    params = { 
        "features__categorical__categorical-features-only__do_predictors" : [ True ],
        "features__categorical__categorical-features-only__do_numerical" : [ False ],
        "features__categorical__encode-category-bits__categories": [ 'auto' ],
        "features__categorical__encode-category-bits__handle_unknown": [ 'ignore' ],
    }
    if my_args.categorical_missing_strategy:
        params["features__categorical__missing-data__strategy"] = [ 'most_frequent' ]
    return params

def make_predictor_params(my_args):
    p1 = make_numerical_predictor_params(my_args)
    p2 = make_categorical_predictor_params(my_args)
    p1.update(p2)
    return p1

def make_SGD_params(my_args):
    SGD_params = {
        # [ "hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron",
        # "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive" ], 
        "model__loss": [ "hinge" ], 
        "model__penalty": [ "l2" ], # ["l2", "l1", "elasticnet", None],
        "model__alpha": [ 0.0001 ],  # Float in range [0.0, inf)
        "model__l1_ratio": [ 0.15 ],  # Float in range [0.0, 1.0]
        "model__fit_intercept": [ True ], # [True, False],
        "model__max_iter": [ 1000 ],  # Int in range [1, inf)
        "model__tol": [ 1e-3 ],  # Float in range [0.0, inf) or None
        "model__shuffle": [ True ], # [True, False],
        "model__verbose": [ 0 ],  # Int in range [0, inf)
        "model__epsilon": [ 0.1 ],  # Float in range [0.0, inf)
        "model__n_jobs": [ None ],  # -1 (using all processors) or None
        "model__random_state": [ None ],  # Int in range [0, 2**32 - 1] or RandomState instance
        "model__learning_rate": [ "optimal" ], # ["constant", "optimal", "invscaling", "adaptive"],
        "model__eta0": [ 0.0 ],  # Float in range [0.0, inf)
        "model__power_t": [ 0.5 ],  # Float in range (-inf, inf)
        "model__early_stopping": [ False ], # [True, False],
        "model__validation_fraction": [ 0.1 ],  # Float in range (0.0, 1.0)
        "model__n_iter_no_change": [ 5 ],  # Int in range [1, max_iter)
        "model__class_weight": [ None ],  # [None, "balanced"], 
        "model__warm_start": [ False ], # [True, False],
        "model__average": [ False ], # [True, False], Or int in range [1, n_samples]
    }

    return SGD_params

def make_linear_params(my_args):
    linear_params = {
        "model__alpha": [1.0],  # Float in range (0.0, inf)
        "model__fit_intercept": [True],  # [True, False]
        "model__copy_X": [True],  # [True, False]
        "model__max_iter": [None],  # Int in range [1, inf) or None
        "model__tol": [1e-4],  # Float in range [0.0, inf)
        "model__class_weight": [None],  # [None, "balanced", {class_label: weight}]
        "model__solver": ["auto"],  # ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]
        "model__positive": [False],  # [True, False] (Only supported with "lbfgs" solver)
        "model__random_state": [None],  # Int in range [0, 2**32 - 1] or RandomState instance
    }

    return linear_params 

def make_SVM_params(my_args):
    SVC_params = {
        "model__C": [1.0],  # Float, strictly positive
        "model__kernel": ["rbf"],  # ["linear", "poly", "rbf", "sigmoid", "precomputed"] or callable
        "model__degree": [3],  # Int, non-negative
        "model__gamma": ["scale"],  # ["scale", "auto"] or float, non-negative
        "model__coef0": [0.0],  # Float
        "model__shrinking": [True],  # [True, False]
        "model__probability": [False],  # [True, False]
        "model__tol": [1e-3],  # Float
        "model__cache_size": [200],  # Float (in MB)
        "model__class_weight": [None],  # [None, "balanced"] or dict
        "model__verbose": [False],  # [True, False]
        "model__max_iter": [-1],  # Int, -1 for no limit
        "model__decision_function_shape": ["ovr"],  # ["ovo", "ovr"]
        "model__break_ties": [False],  # [True, False]
        "model__random_state": [None],  # Int, RandomState instance, or None
    }

    return SVC_params 

def make_boost_params(my_args):
    '''
    boost_params = {
        "model__loss": ["log_loss"],  # ["log_loss", "exponential"]
        "model__learning_rate": [0.1],  # Float in range [0.0, inf)
        "model__n_estimators": [100],  # Int in range [1, inf)
        "model__subsample": [1.0],  # Float in range (0.0, 1.0]
        "model__criterion": ["friedman_mse"],  # ["friedman_mse", "squared_error"]
        "model__min_samples_split": [2],  # Int in range [2, inf) or float in range (0.0, 1.0]
        "model__min_samples_leaf": [1],  # Int in range [1, inf) or float in range (0.0, 1.0)
        "model__min_weight_fraction_leaf": [0.0],  # Float in range [0.0, 0.5]
        "model__max_depth": [3],  # Int in range [1, inf) or None
        "model__min_impurity_decrease": [0.0],  # Float in range [0.0, inf)
        "model__init": [None],  # Estimator or "zero", default=None
        "model__random_state": [None],  # Int, RandomState instance, or None
        "model__max_features": [None],  # ["sqrt", "log2"], int in range [1, inf), float in range (0.0, 1.0], or None
        "model__verbose": [0],  # Int in range [0, inf)
        "model__max_leaf_nodes": [None],  # Int in range [2, inf) or None
        "model__warm_start": [False],  # [True, False]
        "model__validation_fraction": [0.1],  # Float in range (0.0, 1.0)
        "model__n_iter_no_change": [None],  # Int in range [1, inf) or None
        "model__tol": [1e-4],  # Float in range [0.0, inf)
        "model__ccp_alpha": [0.0],  # Float in range [0.0, inf)
    }
    '''
    boost_params = {
        "model__loss": ["log_loss", "exponential"],  # ["log_loss", "exponential"]
        "model__learning_rate": [0.1, 0.3, 0.5],  # Float in range [0.0, inf)
        "model__n_estimators": [100, 50, 200],  # Int in range [1, inf)
        "model__subsample": [1.0, 0.5, 0.9],  # Float in range (0.0, 1.0]
        "model__criterion": ["friedman_mse", "squared_error"],  # ["friedman_mse", "squared_error"]
        "model__min_samples_split": [2],  # Int in range [2, inf) or float in range (0.0, 1.0]
        "model__min_samples_leaf": [1],  # Int in range [1, inf) or float in range (0.0, 1.0)
        "model__min_weight_fraction_leaf": [0.0],  # Float in range [0.0, 0.5]
        "model__max_depth": [3, None],  # Int in range [1, inf) or None
        "model__min_impurity_decrease": [0.0],  # Float in range [0.0, inf)
        "model__init": [None],  # Estimator or "zero", default=None
        "model__random_state": [None],  # Int, RandomState instance, or None
        "model__max_features": ["sqrt", "log2", None],  # ["sqrt", "log2"], int in range [1, inf), float in range (0.0, 1.0], or None
        "model__verbose": [0],  # Int in range [0, inf)
        "model__max_leaf_nodes": [None],  # Int in range [2, inf) or None
        "model__warm_start": [False],  # [True, False]
        "model__validation_fraction": [0.1],  # Float in range (0.0, 1.0)
        "model__n_iter_no_change": [None],  # Int in range [1, inf) or None
        "model__tol": [1e-4],  # Float in range [0.0, inf)
        "model__ccp_alpha": [0.0, 0.5, 1.0],  # Float in range [0.0, inf)
    }
    return boost_params 

def make_forest_params(my_args):
    '''
    forest_params = {
        "model__n_estimators": [100],  # Int, default=100
        "model__criterion": ["gini"],  # ["gini", "entropy", "log_loss"]
        "model__max_depth": [None],  # Int or None
        "model__min_samples_split": [2],  # Int in range [2, inf) or float in range (0.0, 1.0]
        "model__min_samples_leaf": [1],  # Int in range [1, inf) or float in range (0.0, 1.0]
        "model__min_weight_fraction_leaf": [0.0],  # Float in range [0.0, 0.5]
        "model__max_features": ["sqrt"],  # ["sqrt", "log2", None] or int or float in range (0.0, 1.0]
        "model__max_leaf_nodes": [None],  # Int in range [2, inf) or None
        "model__min_impurity_decrease": [0.0],  # Float in range [0.0, inf)
        "model__bootstrap": [True],  # [True, False]
        "model__oob_score": [False],  # [True, False] or callable
        "model__n_jobs": [None],  # Int or None
        "model__random_state": [None],  # Int, RandomState instance, or None
        "model__verbose": [0],  # Int in range [0, inf)
        "model__warm_start": [False],  # [True, False]
        "model__class_weight": [None],  # ["balanced", "balanced_subsample"] or dict or list of dicts
        "model__ccp_alpha": [0.0],  # Float in range [0.0, inf)
        "model__max_samples": [None],  # Float in range (0.0, 1.0] or Int or None
        "model__monotonic_cst": [None],  # None or array-like
    }
    '''
    forest_params = {
        "model__n_estimators": [100, 500, 50],  # Int, default=100
        "model__criterion": ["gini", "entropy", "log_loss"],
        "model__max_depth": [None],  # Int or None
        "model__min_samples_split": [2],  # Int in range [2, inf) or float in range (0.0, 1.0]
        "model__min_samples_leaf": [1],  # Int in range [1, inf) or float in range (0.0, 1.0]
        "model__min_weight_fraction_leaf": [0.0],  # Float in range [0.0, 0.5]
        "model__max_features": ["sqrt", "log2", None],  #  int or float in range (0.0, 1.0]
        "model__max_leaf_nodes": [None],  # Int in range [2, inf) or None
        "model__min_impurity_decrease": [0.0],  # Float in range [0.0, inf)
        "model__bootstrap": [True],  # [True, False]
        "model__oob_score": [False],  # [True, False] or callable
        "model__n_jobs": [-1],  # Int or None
        "model__random_state": [None],  # Int, RandomState instance, or None
        "model__verbose": [0],  # Int in range [0, inf)
        "model__warm_start": [False],  # [True, False]
        "model__class_weight": [None],  # ["balanced", "balanced_subsample"] or dict or list of dicts
        "model__ccp_alpha": [0.0],  # Float in range [0.0, inf)
        "model__max_samples": [None],  # Float in range (0.0, 1.0] or Int or None
        "model__monotonic_cst": [None],  # None or array-like
    }
    return forest_params 

def make_tree_params(my_args):
    tree_params = {
        "model__criterion": [ "entropy" ], # [ "entropy", "gini" ],
        "model__splitter": [ "best" ], # [ "best", "random" ],
        "model__max_depth": [ 1, 2, 3, 4, None ],
        "model__min_samples_split": [ 2 ], # [ 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64 ],
        "model__min_samples_leaf":  [ 1 ],  # [ 0.01, 0.02, 0.04, 0.1 ],
        "model__max_features":  [ None ], # [ "sqrt", "log2", None ],
        "model__max_leaf_nodes": [ None ], # [ 2, 4, 8, 16, 32, 64, None ],
        "model__min_impurity_decrease": [ 0.0 ], # [ 0.0, 0.01, 0.02, 0.04, 0.1, 0.2 ],
    }
    return tree_params

def make_fit_params(my_args):
    params = make_predictor_params(my_args)
    if my_args.model_type == "SGD":
        model_params = make_SGD_params(my_args)
    elif my_args.model_type == "linear":
        model_params = make_linear_params(my_args)
    elif my_args.model_type == "SVM":
        model_params = make_SVM_params(my_args)
    elif my_args.model_type == "boost":
        model_params = make_boost_params(my_args)
    elif my_args.model_type == "forest":
        model_params = make_forest_params(my_args)
    elif my_args.model_type == "tree":
        model_params = make_tree_params(my_args)
    else:
        raise Exception("Unknown model type: {} [SGD, linear, SVM, boost, forest]".format(my_args.model_type))

    params.update(model_params)
    return params

def make_numerical_feature_pipeline(my_args):
    items = []

    items.append(("numerical-features-only", DataFrameSelector(do_predictors=True, do_numerical=True)))

    if my_args.numerical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.numerical_missing_strategy)))
    if my_args.use_polynomial_features:
        items.append(("polynomial-features", sklearn.preprocessing.PolynomialFeatures(degree=my_args.use_polynomial_features)))
    if my_args.use_scaler:
        items.append(("scaler", sklearn.preprocessing.StandardScaler()))
    items.append(("noop", PipelineNoop()))
    
    numerical_pipeline = sklearn.pipeline.Pipeline(items)
    return numerical_pipeline


def make_categorical_feature_pipeline(my_args):
    items = []
    
    items.append(("categorical-features-only", DataFrameSelector(do_predictors=True, do_numerical=False)))

    if my_args.categorical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.categorical_missing_strategy)))
    items.append(("encode-category-bits", sklearn.preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')))

    categorical_pipeline = sklearn.pipeline.Pipeline(items)
    return categorical_pipeline

def make_feature_pipeline(my_args):
    """
    Numerical features and categorical features are usually preprocessed
    differently. We split them out here, preprocess them, then merge
    the preprocessed features into one group again.
    """
    items = []

    items.append(("numerical", make_numerical_feature_pipeline(my_args)))
    items.append(("categorical", make_categorical_feature_pipeline(my_args)))
    pipeline = sklearn.pipeline.FeatureUnion(transformer_list=items)
    return pipeline


def make_fit_pipeline_regression(my_args):
    """
    These are all regression models.
    """
    items = []
    items.append(("features", make_feature_pipeline(my_args)))
    if my_args.model_type == "SGD":
        items.append(("model", sklearn.linear_model.SGDRegressor(max_iter=10000, n_iter_no_change=100, penalty=None))) # verbose=3, 
    elif my_args.model_type == "linear":
        items.append(("model", sklearn.linear_model.LinearRegression()))
    elif my_args.model_type == "SVM":
        items.append(("model", sklearn.svm.SVR()))
    elif my_args.model_type == "boost":
        items.append(("model", sklearn.ensemble.GradientBoostingRegressor()))
    elif my_args.model_type == "forest":
        items.append(("model", sklearn.ensemble.RandomForestRegressor()))
    elif my_args.model_type == "tree":
        items.append(("model", sklearn.tree.DecisionTreeRegressor()))
    else:
        raise Exception("Unknown model type: {} [SGD, linear, SVM, boost, forest]".format(my_args.model_type))

    return sklearn.pipeline.Pipeline(items)

def make_fit_pipeline_classification(my_args):
    """
    These are all classification models.
    """
    items = []
    items.append(("features", make_feature_pipeline(my_args)))
    if my_args.model_type == "SGD":
        items.append(("model", sklearn.linear_model.SGDClassifier(max_iter=10000, n_iter_no_change=100, penalty=None))) # verbose=3, 
    elif my_args.model_type == "linear":
        items.append(("model", sklearn.linear_model.RidgeClassifier()))
    elif my_args.model_type == "SVM":
        items.append(("model", sklearn.svm.SVC(probability=True)))
    elif my_args.model_type == "boost":
        items.append(("model", sklearn.ensemble.GradientBoostingClassifier()))
    elif my_args.model_type == "forest":
        items.append(("model", sklearn.ensemble.RandomForestClassifier()))
    elif my_args.model_type == "tree":
        items.append(("model", sklearn.tree.DecisionTreeClassifier()))
    else:
        raise Exception("Unknown model type: {} [SGD, linear, SVM, boost, forest]".format(my_args.model_type))

    return sklearn.pipeline.Pipeline(items)

def make_fit_pipeline(my_args):
    return make_fit_pipeline_classification(my_args)
