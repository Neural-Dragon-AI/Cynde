import polars as pl
import numpy as np
from typing import Optional, Tuple
from cynde.functional.predict.types import InputConfig,FeatureSet
import os

def convert_utf8_to_enum(df: pl.DataFrame, threshold: float = 0.2) -> pl.DataFrame:
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1 (exclusive).")

    for column in df.columns:
        if df[column].dtype == pl.Utf8 and len(df[column]) > 0:
            unique_values = df[column].unique()
            unique_ratio = len(unique_values) / len(df[column])

            if unique_ratio <= threshold:
                enum_dtype = pl.Enum(unique_values.to_list())
                df = df.with_columns(df[column].cast(enum_dtype))
            else:
                print(f"Column '{column}' has a high ratio of unique values ({unique_ratio:.2f}). Skipping conversion to Enum.")
        elif df[column].dtype == pl.Utf8 and len(df[column]) == 0:
            print(f"Column '{column}' is empty. Skipping conversion to Enum.")

    return df

def convert_enum_to_physical(df: pl.DataFrame) -> pl.DataFrame:
    df_physical = df.with_columns(
        [pl.col(col).to_physical() for col in df.columns if df[col].dtype == pl.Enum]
    )
    return df_physical

def check_add_cv_index(df:pl.DataFrame,strict:bool=False) -> Optional[pl.DataFrame]:
    if "cv_index" not in df.columns and not strict:
        df = df.with_row_index(name="cv_index")
    elif "cv_index" not in df.columns and strict:
        raise ValueError("cv_index column not found in the DataFrame.")
    return df

def preprocess_inputs(df: pl.DataFrame, input_config: InputConfig):
    """ Saves .parquet for each feature set in input_config """
    save_folder = input_config.save_folder
    for feature_set in input_config.feature_sets:
        feature_set_df = df.select(pl.col("cv_index"),pl.col("target"),[column_name for column_name in feature_set.column_names()])
        save_name = feature_set.joined_names()
        save_path = os.path.join(save_folder, f"{save_name}.parquet")
        feature_set_df.write_parquet(save_path)

def load_preprocessed_features(feature_fold_path:str) -> Tuple[pl.DataFrame,pl.DataFrame,pl.DataFrame]:
    """ Loads the the train,val and test df for a specific feature set fold """
    pass

def validate_preprocessed_inputs(input_config:InputConfig) -> None:
    """ Validates the preprocessed input config checking if the .parquet files are present """
    path = input_config.save_folder
    for feature_set in input_config.feature_sets:
        save_name = feature_set.joined_names()
        save_path = os.path.join(path, f"{save_name}.parquet")
        if not os.path.exists(save_path):
            raise ValueError(f"Preprocessed feature set '{save_name}' not found at '{save_path}'.")
