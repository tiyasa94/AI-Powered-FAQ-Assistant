"""
This module contains utility functions to preprocess an HR Q&A dataset for ingestion
into a vector database. It handles:
- Loading raw data
- Cleaning and normalizing text
- Tagging questions into HR-related categories
"""

import pandas as pd
import os

def load_data(file_path):
    """Load the input CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def clean_and_normalize(df):
    """Strip whitespace and normalize casing of questions and answers."""
    df["Question"] = df["Question"].str.strip().str.capitalize()
    df["Answer"] = df["Answer"].str.strip()
    return df

def categorize_question(q):
    """Assign a category label based on keywords in the question."""
    q = q.lower()
    if "leave" in q:
        return "Leave Policy"
    elif "pay" in q or "salary" in q:
        return "Payroll"
    elif "resignation" in q or "notice" in q:
        return "Resignation"
    elif "performance" in q:
        return "Performance"
    elif "remote" in q or "work-from-home" in q:
        return "Remote Work"
    else:
        return "General HR"

def add_categories(df):
    """Apply the categorize_question function to all rows and create a 'Category' column."""
    df["Category"] = df["Question"].apply(categorize_question)
    return df

def preprocess_dataframe(file_path):
    """
    Complete preprocessing pipeline:
    - Load CSV
    - Clean text
    - Add categories

    Returns:
        Processed pandas DataFrame
    """
    df = load_data(file_path)
    df = clean_and_normalize(df)
    df = add_categories(df)
    return df

def save_data(df, output_path):
    """Save the processed DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
