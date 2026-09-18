"""
Script to validate the knowledge base CSV.
Checks for formatting issues, duplicates, and completeness.
"""

import csv
import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "knowledge_base.csv")


def validate_knowledge_base():
    """Validate the knowledge base CSV."""
    print("=" * 60)
    print("🔍 VALIDATING KNOWLEDGE BASE")
    print("=" * 60)

    # Load CSV
    try:
        df = pd.read_csv(DATA_PATH, quoting=1)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False

    # Check columns
    if list(df.columns) != ['question', 'answer']:
        print(f"❌ Invalid columns: {list(df.columns)}")
        return False
    print(f"✅ Columns: {list(df.columns)}")

    # Check total entries
    total = len(df)
    print(f"✅ Total entries: {total}")

    # Check for nulls
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        print(f"❌ Found {nulls} null values")
        return False
    print("✅ No null values")

    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"⚠️ Found {duplicates} duplicate rows")
    else:
        print("✅ No duplicates")

    # Check question format
    questions_without_mark = df[~df['question'].str.endswith('?')]
    if len(questions_without_mark) > 0:
        print(f"⚠️ {len(questions_without_mark)} questions without question mark")

    # Check answer distribution
    answer_counts = df['answer'].value_counts()
    print(f"\n📊 Answer distribution:")
    for answer, count in answer_counts.items():
        print(f"   {answer}: {count}")

    print("\n" + "=" * 60)
    print("✅ Knowledge base validation complete!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    validate_knowledge_base()