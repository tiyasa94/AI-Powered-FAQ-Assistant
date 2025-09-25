"""
document_loader.py

This module provides utility functions to preprocess an HR Q&A dataset
for ingestion into a vector database. It handles:
- Loading raw data from a CSV file.
- Cleaning and normalizing questions and answers.
"""

import pandas as pd
import os


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the input CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist at the provided path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip extra whitespace and normalize casing for question and answer text.

    Args:
        df (pd.DataFrame): DataFrame containing 'Question' and 'Answer' columns.

    Returns:
        pd.DataFrame: DataFrame with cleaned and normalized text.
    """
    df["Question"] = df["Question"].str.strip().str.capitalize()
    df["Answer"] = df["Answer"].str.strip()
    return df


def preprocess_dataframe(file_path: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline for HR Q&A data:
    1. Load the CSV dataset.
    2. Clean and normalize question and answer text.

    Args:
        file_path (str): Path to the raw input CSV file.

    Returns:
        pd.DataFrame: Fully processed and ready-to-use DataFrame.
    """
    df = load_data(file_path)
    df = clean_and_normalize(df)
    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the processed DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): Processed HR Q&A dataset.
        output_path (str): Path to save the processed CSV file.

    Returns:
        None
    """
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
