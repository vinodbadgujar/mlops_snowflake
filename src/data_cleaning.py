# src/data_cleaning.py

from src.snowflake_client import save_cleaned_dataframe
from src.snowflake_client import fetch_table_as_dataframe
import pandas as pd


def drop_single_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = df.columns[df.nunique() == 1]
    if len(cols_to_drop) > 0:
        print(f"Columns to be dropped: {list(cols_to_drop)}")
        df = df.drop(columns=cols_to_drop)
    else:
        print("No single-value columns to drop.")

    return df

def drop_rows_with_missing_target(df: pd.DataFrame, target_column: str = "FS") -> pd.DataFrame:
    missing_count = df[target_column].isna().sum()
    if missing_count > 0:
        print(f"Number of rows to be dropped (missing {target_column}): {missing_count}")
        df = df.dropna(subset=[target_column])
    else:
        print(f"No missing values in '{target_column}'. No rows dropped.")
    return df


#high missing values column drop
def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    missing_percentage = df.isnull().mean()
    cols_to_drop = missing_percentage[missing_percentage > threshold].index
    print(f"Dropping columns with > {threshold*100}% missing values: {list(cols_to_drop)}")
    return df.drop(columns=cols_to_drop)

#remove duplicates
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Removed {before - after} duplicate rows.")
    return df


##impute missing values
def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    for col in df.columns:
        if df[col].dtype in ['float64', 'int64', 'int32', 'int16', 'int8']:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                print(f"Imputed missing values in numeric column '{col}' with median: {median_value}")
        elif df[col].dtype == 'object':
            if df[col].isnull().any():
                mode_value = df[col].mode().iloc[0]
                df[col] = df[col].fillna(mode_value)
                print(f"Imputed missing values in categorical column '{col}' with mode: {mode_value}")
    return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    print("Optimized numeric datatypes.")
    return df


if __name__ == "__main__":
    table_name = "BRONZE"

    print(f"Connecting to Snowflake and fetching table: {table_name}")
    df = fetch_table_as_dataframe(table_name)

    df = drop_single_value_columns(df)
    print(f"DataFrame shape after dropping single value columns: {df.shape}")

    df = drop_rows_with_missing_target(df, "FS")

    df = drop_high_missing_columns(df, threshold=0.5)
    print(f"DataFrame shape after dropping high missing columns: {df.shape}")

    df = impute_missing_values(df)
    df = remove_duplicates(df)

    df = optimize_dtypes(df)
    save_cleaned_dataframe(df, "SILVER")
    print("Cleaned data saved to Snowflake table: SILVER")