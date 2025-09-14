#!/usr/bin/env python
import pandas as pd
import os

# Define the directory containing the TSV files
tsv_dir = "../data/input_dataset/"

# Load the training data for exploration
train_file = "Arabic LLMs Hallucination-OSACT2024-Train.txt"
train_path = os.path.join(tsv_dir, train_file)

print("=== Data Structure Exploration ===")
print(f"Loading file: {train_file}")

try:
    df = pd.read_csv(train_path, sep='\t', engine='python', quoting=3)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\n=== Sample Data ===")
    print(df.head(10))
    
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    print("\n=== Unique Values Analysis ===")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")
        if unique_count < 20:  # Show unique values for categorical columns
            print(f"  Values: {df[col].unique()}")
    
    print("\n=== Readability Distribution ===")
    print(df['readability'].value_counts().sort_index())
    
    print("\n=== Model Distribution ===")
    print(df['model'].value_counts())
    
    print("\n=== Label Distribution ===")
    print(df['label'].value_counts())
    
    print("\n=== Word POS Examples ===")
    print(df['word_pos'].head(20).tolist())
    
    print("\n=== Claim Length Analysis ===")
    df['claim_length'] = df['claim'].str.len()
    print(f"Average claim length: {df['claim_length'].mean():.2f}")
    print(f"Min claim length: {df['claim_length'].min()}")
    print(f"Max claim length: {df['claim_length'].max()}")
    print(f"Claim length distribution:")
    print(df['claim_length'].describe())
    
except Exception as e:
    print(f"Error: {e}") 