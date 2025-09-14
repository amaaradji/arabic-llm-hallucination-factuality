#!/usr/bin/env python
import pandas as pd
import numpy as np
import re
import os
from collections import Counter

# Define the directory containing the TSV files
tsv_dir = "../data/input_dataset/"
output_dir = "../feature_engineered_data/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def extract_pos_tag(word_pos_string):
    """Extract the main POS tag from word_pos string"""
    if pd.isna(word_pos_string):
        return "other"
    word_pos_string = str(word_pos_string)
    parts = word_pos_string.split("_")
    for part in parts:
        if part in ["noun", "verb", "adj", "adv", "prep", "conj", "interj", "pseudo"]:
            return part
    return "other"

def count_arabic_characters(text):
    """Count Arabic characters in the text"""
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    return len(arabic_pattern.findall(text))

def count_diacritics(text):
    """Count Arabic diacritics (tashkeel) in the text"""
    diacritics_pattern = re.compile(r'[\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]')
    return len(diacritics_pattern.findall(text))

def count_punctuation(text):
    """Count punctuation marks in the text"""
    punctuation_pattern = re.compile(r'[،؛؟!\.،:؛]')
    return len(punctuation_pattern.findall(text))

def count_numbers(text):
    """Count numbers in the text"""
    number_pattern = re.compile(r'\d+')
    return len(number_pattern.findall(text))

def count_english_words(text):
    """Count English words in the text"""
    english_pattern = re.compile(r'\b[a-zA-Z]+\b')
    return len(english_pattern.findall(text))

def calculate_word_diversity(text):
    """Calculate word diversity (unique words / total words)"""
    words = text.split()
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)

def extract_linguistic_features(text):
    """Extract various linguistic features from Arabic text"""
    features = {}
    
    # Basic text statistics
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    features['char_count'] = len(text.replace(' ', ''))
    features['avg_word_length'] = features['char_count'] / features['word_count'] if features['word_count'] > 0 else 0
    
    # Arabic-specific features
    features['arabic_char_count'] = count_arabic_characters(text)
    features['diacritics_count'] = count_diacritics(text)
    features['punctuation_count'] = count_punctuation(text)
    features['number_count'] = count_numbers(text)
    features['english_word_count'] = count_english_words(text)
    
    # Linguistic ratios
    features['arabic_char_ratio'] = features['arabic_char_count'] / features['char_count'] if features['char_count'] > 0 else 0
    features['diacritics_ratio'] = features['diacritics_count'] / features['arabic_char_count'] if features['arabic_char_count'] > 0 else 0
    features['punctuation_ratio'] = features['punctuation_count'] / features['word_count'] if features['word_count'] > 0 else 0
    
    # Text complexity features
    features['word_diversity'] = calculate_word_diversity(text)
    features['sentence_count'] = len(re.split(r'[.!?؟،]', text)) - 1  # Approximate sentence count
    
    return features

def create_model_features(model_name):
    """Create model-specific features"""
    features = {}
    
    # Model encoding
    features['model_gpt4'] = 1 if model_name == 'GPT4' else 0
    features['model_chatgpt'] = 1 if model_name == 'ChatGPT' else 0
    
    return features

def create_readability_features(readability_level):
    """Create readability-specific features"""
    features = {}
    
    # Readability level encoding
    features['readability_level'] = readability_level
    
    # Readability categories
    features['readability_easy'] = 1 if readability_level <= 2 else 0
    features['readability_medium'] = 1 if 2 < readability_level <= 3 else 0
    features['readability_hard'] = 1 if readability_level >= 4 else 0
    
    return features

def create_pos_features(pos_tag):
    """Create POS-specific features"""
    features = {}
    
    # Main POS categories
    pos_categories = ['noun', 'verb', 'adj', 'adv', 'prep', 'conj', 'interj', 'pseudo', 'other']
    
    for pos in pos_categories:
        features[f'pos_{pos}'] = 1 if pos_tag == pos else 0
    
    return features

def create_label_features(label):
    """Create label-specific features"""
    features = {}
    
    # Label encoding
    features['label_fc'] = 1 if label == 'FC' else 0  # Factually Correct
    features['label_nf'] = 1 if label == 'NF' else 0  # Non-Factual
    features['label_fi'] = 1 if label == 'FI' else 0  # Factually Incorrect
    
    return features

def engineer_features(df):
    """Main feature engineering function"""
    print("Starting feature engineering...")
    
    # Initialize lists to store engineered features
    engineered_features = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{len(df)}")
        
        # Extract features from claim text
        text_features = extract_linguistic_features(row['claim'])
        
        # Extract POS features
        pos_tag = extract_pos_tag(row['word_pos'])
        pos_features = create_pos_features(pos_tag)
        
        # Extract model features
        model_features = create_model_features(row['model'])
        
        # Extract readability features
        readability_features = create_readability_features(row['readability'])
        
        # Extract label features
        label_features = create_label_features(row['label'])
        
        # Combine all features
        combined_features = {
            'claim_id': row['claim_id'],
            **text_features,
            **pos_features,
            **model_features,
            **readability_features,
            **label_features
        }
        
        engineered_features.append(combined_features)
    
    # Convert to DataFrame
    engineered_df = pd.DataFrame(engineered_features)
    
    print(f"Feature engineering complete. Shape: {engineered_df.shape}")
    print(f"Features created: {list(engineered_df.columns)}")
    
    return engineered_df

def main():
    """Main execution function"""
    files_to_process = {
        "train": "Arabic LLMs Hallucination-OSACT2024-Train.txt",
        "dev": "Arabic LLMs Hallucination-OSACT2024-Dev.txt",
        "test": "Arabic LLMs Hallucination-OSACT2024-Test.txt"
    }
    
    for dataset_name, file_name in files_to_process.items():
        print(f"\n=== Processing {dataset_name} dataset ===")
        
        # Load data
        file_path = os.path.join(tsv_dir, file_name)
        df = pd.read_csv(file_path, sep='\t', engine='python', quoting=3)
        
        # Engineer features
        engineered_df = engineer_features(df)
        
        # Save engineered features
        output_file = os.path.join(output_dir, f"{dataset_name}_engineered_features.csv")
        engineered_df.to_csv(output_file, index=False)
        print(f"Saved engineered features to: {output_file}")
        
        # Print feature summary
        print(f"\nFeature summary for {dataset_name}:")
        print(f"Original features: {len(df.columns)}")
        print(f"Engineered features: {len(engineered_df.columns)}")
        print(f"Total features: {len(engineered_df.columns)}")
        
        # Show sample of engineered features
        print(f"\nSample engineered features for {dataset_name}:")
        print(engineered_df.head(3))

if __name__ == "__main__":
    main() 