def load_excel_file(file_path):
    import pandas as pd
    return pd.read_excel(file_path)

def save_to_excel(df, file_path):
    df.to_excel(file_path, index=False)

def convert_to_numeric(series):
    return pd.to_numeric(series, errors='coerce')

def filter_positive_values(df, columns):
    for column in columns:
        df = df[df[column] > 0]
    return df

def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def split_column(df, column_name, separator=','):
    return df[column_name].str.split(separator, expand=True)