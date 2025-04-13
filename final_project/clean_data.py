#!/usr/bin/env python3

import pandas as pd

def balance_dataset(input_file, output_file):
    # Read the original dataset
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Count positive and negative cases
    positive_cases = df[df['diabetes'] == 1]
    negative_cases = df[df['diabetes'] == 0]
    
    print(f"Original dataset:")
    print(f"Positive cases: {len(positive_cases)}")
    print(f"Negative cases: {len(negative_cases)}")
    
    # Randomly sample from negative cases to match positive cases
    balanced_negative = negative_cases.sample(n=len(positive_cases), random_state=42)
    
    # Combine positive and balanced negative cases
    balanced_df = pd.concat([positive_cases, balanced_negative])
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalanced dataset:")
    print(f"Total cases: {len(balanced_df)}")
    print(f"Positive cases: {len(balanced_df[balanced_df['diabetes'] == 1])}")
    print(f"Negative cases: {len(balanced_df[balanced_df['diabetes'] == 0])}")
    
    # Save the balanced dataset
    print(f"\nSaving balanced dataset to {output_file}...")
    balanced_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Define paths
    input_file = "data/diabetes_dataset.csv"
    output_file = "data/balanced_diabetes_dataset.csv"
    
    # Balance the dataset
    balance_dataset(input_file, output_file) 