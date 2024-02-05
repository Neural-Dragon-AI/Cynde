import time
from typing import List, Union, Any, Optional
import polars as pl
from cynde.models.embedders import  get_embedding_list
from openai import Client

def create_embedding_df_from_column(
    df: pl.DataFrame, 
    column_name: str, 
    model: str = "text-embedding-3-small", 
    batch_size: int = 100, 
    client: Optional[Client] = None
) -> pl.DataFrame:
    """
    Generates embeddings for unique values in a specified column of a DataFrame and returns a new DataFrame with these embeddings.

    Parameters:
    - df (pl.DataFrame): The source DataFrame.
    - column_name (str): The name of the column for which embeddings will be generated.
    - model (str): The model identifier used for generating embeddings. Default is "text-embedding-3-small".
    - batch_size (int): The number of items to process in each batch request to the embedding model.
    - client (Optional[Client]): An optional OpenAI Client instance for making requests to the embedding API.

    Returns:
    - pl.DataFrame: A new DataFrame containing the original unique column values and their corresponding embeddings.
    """
    print(f"Creating embeddings for column {column_name}")
    #chek that column type is pl.Utf8
    if df[column_name].dtype != pl.Utf8:
        raise ValueError(f"Column {column_name} must be of type Utf8")
    column = df[column_name].unique()
    column_list = column.to_list()
    embeddings = get_embedding_list(column_list, model=model, batch_size=batch_size, client=client)
    embedding_column_name = f"{column_name}_{model}_embeddings"
    embeddings_df = pl.DataFrame({column_name: column_list, embedding_column_name: embeddings})
    return embeddings_df


def embed_columns(
    df: pl.DataFrame, 
    column_names: List[Union[str, List[str]]], 
    model: str = "text-embedding-3-small", 
    batch_size: int = 100, 
    separator: str = " ", 
    client: Optional[Client] = None
) -> pl.DataFrame:
    """
    Processes specified columns in a DataFrame to generate embeddings, merging columns if necessary, and appends the embeddings as new columns.

    Parameters:
    - df (pl.DataFrame): The source DataFrame.
    - column_names (List[Union[str, List[str]]]): A list of column names or lists of column names to be merged before generating embeddings.
    - model (str): The model identifier used for generating embeddings. Default is "text-embedding-3-small".
    - batch_size (int): The number of items to process in each batch request to the embedding model.
    - separator (str): The separator string used when merging columns. Default is a single space " ".
    - client (Optional[Client]): An optional OpenAI Client instance for making requests to the embedding API.

    Returns:
    - pl.DataFrame: The input DataFrame with new columns added for the embeddings of the specified (and potentially merged) columns.
    """
    new_column_names = []
    for column_name in column_names:
        if isinstance(column_name, list):
            merged_name = "_".join(column_name)
            expression = pl.col(column_name[0]).cast(pl.Utf8)
            for name in column_name[1:]:
                expression += separator + pl.col(name).cast(pl.Utf8)
            expression = expression.alias(merged_name)
            new_column_names.append(merged_name)
            print(f"Merged columns {[col for col in column_name]} into {merged_name}")
            df = df.with_columns(expression)
        else:
            new_column_names.append(column_name)

    for column_name in new_column_names:
        embeddings_df = create_embedding_df_from_column(df, column_name, model=model, batch_size=batch_size, client=client)
        df = df.join(embeddings_df, on=column_name, how="left")
    return df
