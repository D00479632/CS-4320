#!/usr/bin/env python3

import sys
import argparse
import logging
import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import math

def get_data(filename):
    """
    Reads a CSV file into a pandas DataFrame.
    Assumes column 0 is the instance index stored in the
    csv file. If no such column exists, remove the
    index_col=0 parameter.
    """
    data = pd.read_csv(filename, index_col=0)  # Read CSV and set first column as index
    return data

def get_basename(filename):
    """
    Extracts the base name (file name without directory and extension) from a given file path.
    Logs the root, extension, directory, and base name for debugging purposes.
    """
    root, ext = os.path.splitext(filename)  
    dirname, basename = os.path.split(root)  
    logging.info("root: {}  ext: {}  dirname: {}  basename: {}".format(root, ext, dirname, basename))
    return basename  

def get_feature_and_label_names(my_args, data):
    """
    Retrieves the feature and label column names based on user arguments and the DataFrame.
    If no features are specified, it defaults to all non-label columns.
    Returns a list of feature names and the label name.
    """
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

    # No features specified, so add all non-labels
    if len(features) == 0:
        for feature_column in data.columns:
            if feature_column != label:
                features.append(feature_column)  

    return features, label  

def display_feature_histograms(my_args, data):
    """
    Displays a histogram for every feature and the label, if identified.
    Creates a separate figure for each feature and saves it as a PDF.
    """
    feature_columns, label_column = get_feature_and_label_names(my_args, data)

    n_max = 1  # Initialize n_max to find the maximum y-value
    # First pass to determine the maximum y-value for all features
    for feature_column in feature_columns:
        if feature_column in data.columns:
            col_df = data[[feature_column]].dropna()  # Drop NaN values for the feature
            max_count = col_df[feature_column].value_counts().max()  # Get the maximum count directly
            if max_count > n_max:
                n_max = max_count  # Update n_max

    # Second pass to create and save each histogram
    for feature_column in feature_columns:
        if feature_column in data.columns:
            # Create a new figure for each feature
            fig = plt.figure(figsize=(6.5, 9))
            fig.suptitle(f"Histogram of {feature_column}")
            ax = fig.add_subplot(1, 1, 1)
            ax.set_yscale("log")
            col_df = data[[feature_column]].dropna()
            ax.hist(col_df[feature_column], bins=20)
            ax.set_xlabel(feature_column)
            ax.locator_params(axis='x', tight=True, nbins=5)
            ax.set_ylim(bottom=1.0, top=n_max)  # Set y-axis limits

            basename = get_basename(my_args.data_file)
            figure_name = f"plots/{basename}-histogram-{feature_column}.pdf"
            fig.savefig(figure_name)
            plt.close(fig)
        else:
            logging.warn("feature_column: '{}' not in data.columns: {}".format(feature_column, data.columns))

    if label_column:
        # Create and save histogram for the label column
        fig = plt.figure(figsize=(6.5, 9))
        fig.suptitle(f"Histogram of {label_column}")
        ax = fig.add_subplot(1, 1, 1)
        ax.set_yscale("log")
        col_df = data[[label_column]].dropna()
        ax.hist(col_df[label_column], bins=20)
        ax.set_xlabel(label_column)
        ax.locator_params(axis='x', tight=True, nbins=5)
        ax.set_ylim(bottom=1.0, top=n_max)  # Set y-axis limits

        basename = get_basename(my_args.data_file)
        figure_name = f"plots/{basename}-histogram-{label_column}.pdf"
        fig.savefig(figure_name)
        plt.close(fig)

    return


def display_label_vs_features(my_args, data):
    """
    Displays a scatter plot of label vs feature for every feature and the label, if identified.
    Creates a separate figure for each feature and saves it as a PDF.
    """
    feature_columns, label_column = get_feature_and_label_names(my_args, data)  

    # Calculate y_max for scatter plots
    y_max = 0
    if label_column in data.columns:
        y_max = data[label_column].max()  # Get max value for label column

    # Create scatter plots with consistent y-axis limits
    for feature_column in feature_columns:
        if feature_column in data.columns:
            fig = plt.figure(figsize=(6.5, 9))  
            fig.suptitle(f"Label vs. {feature_column}")  
            col_df = data[[feature_column, label_column]].dropna()
            ax = fig.add_subplot(1, 1, 1)  
            ax.scatter(col_df[feature_column], col_df[label_column], s=1)  
            ax.set_xlabel(feature_column)  
            ax.set_ylabel(label_column)  
            ax.locator_params(axis='both', tight=True, nbins=5)  
            ax.set_ylim(bottom=0, top=y_max)  # Set consistent y-axis limits

            basename = get_basename(my_args.data_file)  
            figure_name = f"plots/{basename}-scatter-{feature_column}.pdf"  
            fig.savefig(figure_name)  
            plt.close(fig)  

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Create Data Plots')  
    parser.add_argument('action', default='label-vs-features',
                        choices=["label-vs-features", "feature-histograms", "all"], 
                        nargs='?', help="desired display action")  
    parser.add_argument('--data-file', '-d', default="data/train.csv", type=str, help="csv file of data to display")  
    parser.add_argument('--features', '-f', default=None, action="extend", nargs="+", type=str,
                        help="column names for features")  
    parser.add_argument('--label', '-l', default="SalePrice", type=str, help="column name for label")  # Define label argument

    my_args = parser.parse_args(argv[1:])  

    return my_args  

def main(argv):
    my_args = parse_args(argv)  
    logging.basicConfig(level=logging.WARN)  

    filename = my_args.data_file  
    if os.path.exists(filename) and os.path.isfile(filename):  
        data = get_data(filename)  

        if my_args.action in ("all", "label-vs-features"):
            display_label_vs_features(my_args, data)  
        if my_args.action in ("all", "feature-histograms"):
            display_feature_histograms(my_args, data) 

    else:
        print(filename + " doesn't exist, or is not a file.")  
    
    return

if __name__ == "__main__":
    main(sys.argv)  