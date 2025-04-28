# src/snowflake_client.py

from src.config import SNOWFLAKE_CONFIG
import snowflake.connector
import pandas as pd
from snowflake.connector.pandas_tools import write_pandas

def get_snowflake_connection():
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    return conn

def fetch_table_as_dataframe(table_name: str) -> pd.DataFrame:
    conn = get_snowflake_connection()
    cur = conn.cursor()
    try:
        query = f"SELECT * FROM {table_name};"
        cur.execute(query)
        df = cur.fetch_pandas_all()
    finally:
        cur.close()
        conn.close()
    return df

def save_cleaned_dataframe(df: pd.DataFrame, new_table_name: str) -> None:
    conn = get_snowflake_connection()
    try:
        print(f"Saving cleaned data to new Snowflake table: {new_table_name}")
        # Write to Snowflake
        df = df.reset_index(drop=True)
        success, nchunks, nrows, _ = write_pandas(
            conn=conn,
            df=df,
            table_name=new_table_name,
            quote_identifiers=False,
            overwrite=True,  
            auto_create_table=True,
        )
        if success:
            print(f"Successfully saved {nrows} rows to '{new_table_name}'")
        else:
            print(f"Failed to save data to '{new_table_name}'")
    finally:
        conn.close()

if __name__ == "__main__":
    conn = get_snowflake_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(*) FROM BRONZE;
    """)

    row_count = cur.fetchone()[0]
    print(f"Number of rows in table: {row_count}")

    cur.close()
    conn.close()