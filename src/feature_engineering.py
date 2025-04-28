# src/feature_engineering.py

import pandas as pd
from src.snowflake_client import get_snowflake_connection, save_cleaned_dataframe , fetch_table_as_dataframe


def extract_unit_type(df: pd.DataFrame, cat_col: str = "SERIALNO") -> pd.DataFrame:
    df['UnitType'] = df[cat_col].apply(
        lambda x: 'HU' if 'HU' in str(x) else ('GQ' if 'GQ' in str(x) else 'Unknown')
    )
    print(f"Extracted unit type from '{cat_col}' column.")
    return df

def one_hot_encode_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
    dummies = dummies.astype(int)
    dummies.columns = dummies.columns.str.replace('.', '_', regex=False)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=[column])
    print(f"One-hot encoded column: {column}")
    print(f"New columns created: {list(dummies.columns)}")
    return df

#Adjust housing dollars based on inflation factor
def adjust_housing_dollars(df: pd.DataFrame, adjhsg_column: str = 'ADJHSG') -> pd.DataFrame:
    if adjhsg_column not in df.columns:
        raise ValueError(f"Adjustment factor column '{adjhsg_column}' not found in the DataFrame.")

    columns_to_adjust = ['CONP', 'ELEP', 'INSP', 'SMOCP', 'TAXAMT', 'VALP', 'WATP']
    df['inflation_factor'] = df[adjhsg_column] / 1_000_000

    for col in columns_to_adjust:
        if col in df.columns:
            df[col] = df[col] * df['inflation_factor']
        else:
            print(f"Warning: Column '{col}' not found in the DataFrame, skipping adjustment.")
    df = df.drop(columns=['inflation_factor'])
    df = df.drop(columns=[adjhsg_column])
    print(f"Adjusted housing dollars based on '{adjhsg_column}' inflation factor.")
    return df

def adjust_income_for_inflation(df: pd.DataFrame, adjinc_column: str = 'ADJINC') -> pd.DataFrame:
    if adjinc_column not in df.columns:
        raise ValueError(f"Adjustment factor column '{adjinc_column}' not found in the DataFrame.")

    columns_to_adjust = ['FINCP', 'HINCP']

    df['inflation_factor'] = df[adjinc_column] / 1_000_000

    for col in columns_to_adjust:
        if col in df.columns:
            df[col] = df[col] * df['inflation_factor']
        else:
            print(f"Warning: Household income column '{col}' not found in the DataFrame, skipping adjustment.")

    df = df.drop(columns=['inflation_factor'])
    df = df.drop(columns=[adjinc_column])
    print(f"Adjusted income based on '{adjinc_column}' inflation factor.")
    return df

def drop_unnecessary_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    df = df.drop(columns=columns_to_drop, errors='ignore')
    print(f"Dropped columns: {columns_to_drop}")
    return df

def main():
    table_name = "SILVER"      
    new_table_name = "GOLD" 

    print(f"Fetching cleaned table: {table_name}")
    df = fetch_table_as_dataframe(table_name)
    print(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
    df = extract_unit_type(df, cat_col="SERIALNO")
    df = one_hot_encode_column(df, column="UnitType")
    df = one_hot_encode_column(df, column="TYPEHUGQ")
    df = one_hot_encode_column(df, column="ACCESSINET")
    df = one_hot_encode_column(df, column="BATH")
    df = one_hot_encode_column(df, column="FYRBLTP")
    df = one_hot_encode_column(df, column="FWATP")
    df = one_hot_encode_column(df, column="FVEHP")
    df = one_hot_encode_column(df, column="FVALP")
    df = one_hot_encode_column(df, column="FTENP")
    df = one_hot_encode_column(df, column="FTELP")
    df = one_hot_encode_column(df, column="FTAXP")
    df = one_hot_encode_column(df, column="FTABLETP")
    df = one_hot_encode_column(df, column="FSTOVP")
    df = one_hot_encode_column(df, column="FSMXSP")
    df = one_hot_encode_column(df, column="FSMXHP")
    df = one_hot_encode_column(df, column="FSMP")
    df = one_hot_encode_column(df, column="FSMOCP")
    df = one_hot_encode_column(df, column="FSMARTPHONP")
    df = one_hot_encode_column(df, column="FSINKP")
    df = one_hot_encode_column(df, column="FSATELLITEP")
    df = one_hot_encode_column(df, column="FRWATP")
    df = one_hot_encode_column(df, column="FRNTP")
    df = one_hot_encode_column(df, column="FRNTMP")
    df = one_hot_encode_column(df, column="FRMSP")
    df = one_hot_encode_column(df, column="FREFRP")
    df = one_hot_encode_column(df, column="FPLMP")
    df = one_hot_encode_column(df, column="FOTHSVCEXP")
    df = one_hot_encode_column(df, column="FMVP")
    df = one_hot_encode_column(df, column="FMRGXP")
    df = one_hot_encode_column(df, column="FMRGTP")
    df = one_hot_encode_column(df, column="FMRGP")
    df = one_hot_encode_column(df, column="FMRGIP")
    df = one_hot_encode_column(df, column="FMHP")
    df = one_hot_encode_column(df, column="FLAPTOPP")
    df = one_hot_encode_column(df, column="FKITP")
    df = one_hot_encode_column(df, column="FINSP")
    df = one_hot_encode_column(df, column="FHISPEEDP")
    df = one_hot_encode_column(df, column="FHINCP")
    df = one_hot_encode_column(df, column="FHFLP")
    df = one_hot_encode_column(df, column="FGRNTP")
    df = one_hot_encode_column(df, column="FGASP")
    df = one_hot_encode_column(df, column="FFULP")
    df = one_hot_encode_column(df, column="FFSP")
    df = one_hot_encode_column(df, column="FFINCP")
    df = one_hot_encode_column(df, column="FELEP")
    df = one_hot_encode_column(df, column="FDIALUPP")
    df = one_hot_encode_column(df, column="FCONP")
    df = one_hot_encode_column(df, column="FCOMPOTHXP")
    df = one_hot_encode_column(df, column="FBROADBNDP")
    df = one_hot_encode_column(df, column="FBLDP")
    df = one_hot_encode_column(df, column="FBDSP")
    df = one_hot_encode_column(df, column="FBATHP")
    df = one_hot_encode_column(df, column="FAGSP")
    df = one_hot_encode_column(df, column="FACRP")
    df = one_hot_encode_column(df, column="FACCESSP")
    df = one_hot_encode_column(df, column="SVAL")
    df = one_hot_encode_column(df, column="SRNT")
    df = one_hot_encode_column(df, column="R18")
    df = one_hot_encode_column(df, column="PSF")
    df = one_hot_encode_column(df, column="PLM")  
    df = one_hot_encode_column(df, column="PARTNER")
    df = one_hot_encode_column(df, column="NR")
    df = one_hot_encode_column(df, column="NPP")
    df = one_hot_encode_column(df, column="MULTG")
    df = one_hot_encode_column(df, column="LNGI")
    df = one_hot_encode_column(df, column="KIT")
    df = one_hot_encode_column(df, column="HUGCL")
    df = one_hot_encode_column(df, column="FPARC")
    df = one_hot_encode_column(df, column="WATFP")
    df = one_hot_encode_column(df, column="TEN")
    df = one_hot_encode_column(df, column="TEL")
    df = one_hot_encode_column(df, column="TABLET")
    df = one_hot_encode_column(df, column="STOV")
    df = one_hot_encode_column(df, column="SMARTPHONE")
    df = one_hot_encode_column(df, column="SINK")
    df = one_hot_encode_column(df, column="SATELLITE")
    df = one_hot_encode_column(df, column="RWAT")
    df = one_hot_encode_column(df, column="REFR")
    df = one_hot_encode_column(df, column="OTHSVCEX")
    df = one_hot_encode_column(df, column="MRGX")
    df = one_hot_encode_column(df, column="LAPTOP")
    df = one_hot_encode_column(df, column="HISPEED")
    df = one_hot_encode_column(df, column="GASFP")
    df = one_hot_encode_column(df, column="FULFP")
    df = one_hot_encode_column(df, column="ELEFP")
    df = one_hot_encode_column(df, column="DIALUP")
    df = one_hot_encode_column(df, column="COMPOTHX")
    df = one_hot_encode_column(df, column="BROADBND")


    df = adjust_housing_dollars(df, adjhsg_column="ADJHSG")
    df = adjust_income_for_inflation(df, adjinc_column="ADJINC")

    weight_variables = ["WGTP"] + [f"WGTP{i}" for i in range(1, 81)]

    columns_to_drop = ["SERIALNO", "PUMA10", "PUMA20", "RESMODE", "HHLDRRAC1P", "HHLDRHISP", "HHLANP", "HHL", "YRBLT"] + weight_variables

    df = drop_unnecessary_columns(df, columns_to_drop)
    save_cleaned_dataframe(df, new_table_name)

if __name__ == "__main__":
    main()

