"""
Data Preprocessing and Text Cleaning Pipeline.

This module handles:
1. Loading the ticket dataset and performing schema validation.
2. Checking and reporting missing values, duplicates, and column distributions.
3. Systematic text cleaning: lowercasing, punctuation normalization,
   special character stripping, and whitespace standardization.
"""

import re
import string
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional


def clean_text(text: str) -> str:
    """
    Cleans and standardizes raw ticket text.

    Steps:
    1. Handle non-string / missing values safely.
    2. Convert to lowercase for vocabulary normalization.
    3. Remove URLs, email addresses, and HTML tags.
    4. Remove punctuation and special symbols, preserving alphanumeric words.
    5. Collapse multiple whitespace characters into a single space and strip.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Remove HTML tags if present
    text = re.sub(r"<.*?>", " ", text)

    # Remove punctuation and special symbols (keep alphanumeric and spaces)
    # Using regex to replace non-alphanumeric with space
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Collapse multiple whitespaces and trim
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads dataset from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df


def validate_and_summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates dataset integrity and produces a structural summary:
    - Shape (rows, columns)
    - Missing values count per column
    - Duplicate count
    - Class counts for category, priority, status
    """
    required_columns = [
        "ticket_id",
        "customer_name",
        "ticket_description",
        "date",
        "priority",
        "status",
        "category",
    ]

    missing_cols = [c for c in required_columns if c not in df.columns]

    missing_values = df.isnull().sum().to_dict()
    duplicates_count = int(df.duplicated(subset=["ticket_description"]).sum())
    total_records = len(df)

    category_counts = (
        df["category"].value_counts().to_dict() if "category" in df.columns else {}
    )
    priority_counts = (
        df["priority"].value_counts().to_dict() if "priority" in df.columns else {}
    )
    status_counts = (
        df["status"].value_counts().to_dict() if "status" in df.columns else {}
    )

    summary = {
        "total_records": total_records,
        "total_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "missing_required_columns": missing_cols,
        "missing_values": missing_values,
        "duplicate_descriptions": duplicates_count,
        "category_counts": category_counts,
        "priority_counts": priority_counts,
        "status_counts": status_counts,
    }
    return summary


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "ticket_description",
    cleaned_column: str = "cleaned_description",
    drop_duplicates: bool = True,
    drop_nulls: bool = True,
) -> pd.DataFrame:
    """
    Preprocesses a pandas DataFrame:
    1. Optionally removes null descriptions.
    2. Optionally drops exact duplicates.
    3. Adds a cleaned text column using `clean_text`.
    """
    processed_df = df.copy()

    if drop_nulls and text_column in processed_df.columns:
        processed_df = processed_df.dropna(subset=[text_column])

    if drop_duplicates and text_column in processed_df.columns:
        processed_df = processed_df.drop_duplicates(subset=[text_column])

    processed_df[cleaned_column] = processed_df[text_column].apply(clean_text)

    # Compute word and character lengths for analytical reference
    processed_df["char_length"] = processed_df[cleaned_column].apply(len)
    processed_df["word_count"] = processed_df[cleaned_column].apply(
        lambda s: len(s.split())
    )

    return processed_df


if __name__ == "__main__":
    import os

    dataset_path = "data/customer_support_ticket_dataset_200.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "customer_support_ticket_dataset_200.csv"

    print("=" * 60)
    print("CUSTOMER SUPPORT TICKET DATASET PREPROCESSING & VALIDATION")
    print("=" * 60)

    df_raw = load_dataset(dataset_path)
    summary = validate_and_summarize_dataset(df_raw)

    print(f"Total Records: {summary['total_records']}")
    print(f"Total Columns: {summary['total_columns']}")
    print(f"Columns: {summary['columns']}")
    print(f"Missing Values: {summary['missing_values']}")
    print(f"Duplicate Descriptions: {summary['duplicate_descriptions']}")
    print(f"\nUnique Categories ({len(summary['category_counts'])}):")
    for cat, count in summary["category_counts"].items():
        print(f"  - {cat:20s}: {count} records")

    print("\nPriority Breakdown:")
    for prio, count in summary["priority_counts"].items():
        print(f"  - {prio:15s}: {count}")

    print("\nStatus Breakdown:")
    for stat, count in summary["status_counts"].items():
        print(f"  - {stat:15s}: {count}")

    df_processed = preprocess_dataframe(df_raw)
    print("\nSample Preprocessed Records (Original -> Cleaned):")
    for _, row in df_processed.head(5).iterrows():
        print(f"  [RAW]   : {row['ticket_description']}")
        print(f"  [CLEAN] : {row['cleaned_description']}")
        print(f"  [META]  : Cat={row['category']}, Words={row['word_count']}, Chars={row['char_length']}\n")
