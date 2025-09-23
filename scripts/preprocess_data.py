'''
Runner script to preprocess the raw faq data.

Example usage:
    python scripts/preprocess_data.py
'''


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.document_loader import preprocess_dataframe, save_data

if __name__ == "__main__":
    INPUT_PATH = "data/raw/faqs.csv"
    OUTPUT_PATH = "data/processed/processed_faqs.csv"

    df = preprocess_dataframe(INPUT_PATH)
    save_data(df, OUTPUT_PATH)
