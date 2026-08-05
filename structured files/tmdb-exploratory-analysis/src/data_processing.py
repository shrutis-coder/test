import pandas as pd
import glob

def load_data(file_pattern):
    """Load data from an Excel file matching the given pattern."""
    excel_file = glob.glob(file_pattern)[0]
    print(f"Loading: {excel_file}")
    df = pd.read_excel(excel_file)
    return df

def clean_data(df):
    """Clean the DataFrame by converting budget and revenue to numeric and filtering out invalid rows."""
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    return df

def save_cleaned_data(df, output_file):
    """Save the cleaned DataFrame to an Excel file."""
    df.to_excel(output_file, index=False)
    print(f"Saved cleaned data to {output_file}")