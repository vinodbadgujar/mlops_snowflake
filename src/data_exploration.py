# src/data_exploration.py

from src.snowflake_client import fetch_table_as_dataframe
import pandas as pd
import os



def get_missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    missing_report = df.isnull().mean().reset_index()
    missing_report.columns = ['column_name', 'missing_percentage']
    missing_report = missing_report.sort_values(by='missing_percentage', ascending=False)
    return missing_report


def get_basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=['number']) 
    stats = numeric_cols.describe().transpose()
    stats.index.name = 'column_name' 
    stats.reset_index(inplace=True) 
    return stats



def get_column_types(df: pd.DataFrame) -> pd.DataFrame:

    types_report = pd.DataFrame(df.dtypes, columns=['dtype']).reset_index()
    types_report.columns = ['column_name', 'dtype']
    return types_report

def save_report(df: pd.DataFrame, report_name: str):
    os.makedirs("outputs", exist_ok=True)
    path = os.path.join("outputs", f"{report_name}.csv")
    df.to_csv(path, index=False)
    print(f"Saved report: {path}")

def get_unique_values_report(df: pd.DataFrame) -> pd.DataFrame:
    unique_report = df.nunique().reset_index()
    unique_report.columns = ['column_name', 'unique_values_count']
    return unique_report


def get_categorical_value_counts(df: pd.DataFrame, threshold: int = 20):
    categorical_cols = [col for col in df.columns if df[col].nunique() <= threshold]
    for col in categorical_cols:
        print(f"\nValue counts for '{col}':")
        print(df[col].value_counts(dropna=False))

def main():
    table_name = "BRONZE"

    print(f"Connecting to Snowflake and fetching table: {table_name}")
    df = fetch_table_as_dataframe(table_name)
    print(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("Missing Values Report:")
    missing_report = get_missing_values_report(df)
    print(missing_report)
    save_report(missing_report, "missing_values_report")

    print("\nBasic Statistics for Numeric Columns:")
    basic_stats = get_basic_statistics(df)
    print(basic_stats)
    save_report(basic_stats, "basic_statistics_report")

    print("\nColumn Data Types:")
    column_types = get_column_types(df)
    print(column_types)
    save_report(column_types, "column_types_report")

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    unique_values = get_unique_values_report(df)
    print("\nUnique Values per Column:")
    print(unique_values)
    save_report(unique_values, "unique_values_report")

    print("\nCategorical Columns Value Counts:")
    get_categorical_value_counts(df)


if __name__ == "__main__":
    main()

 
