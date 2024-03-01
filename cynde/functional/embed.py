import time
from time import perf_counter
import os
from typing import List, Union, Any, Optional
import polars as pl
from cynde.models.embedders import  get_embedding_list
from openai import Client
import json
import tiktoken

import asyncio
from cynde.async_tools.api_request_parallel_processor import process_api_requests_from_file

MAX_INPUT = {
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191
}

def generate_embedding_payloads_from_column(filename: str, df: pl.DataFrame, column_name: str, model_name: str = "text-embedding-3-small"):
    """Generates embedding payloads from a column in a DataFrame and saves them to a JSONL file."""
    # Extract the data from the specified column as a list
    data = df[column_name].to_list()
    
    with open(filename, "w") as f:
        for x in data:
            # Create the payload for each text entry
            payload = {"model": model_name, "input": str(x)}
            
            # Write the payload to the JSONL file
            json_string = json.dumps(payload)
            f.write(json_string + "\n")

    # Read the JSONL file content into a list of dictionaries
    with open(filename, 'r') as f:
        json_list = [json.loads(line) for line in f]

    # Convert the list of dictionaries into a DataFrame
    payloads_df = pl.DataFrame(json_list)

    # Concatenate the original DataFrame column with the generated payloads DataFrame
    result_df = pl.concat([df.select(column_name), payloads_df], how="horizontal")
    
    return result_df

def generate_embedding_batched_payloads_from_column(filename: str, df: pl.DataFrame, column_name: str, model_name: str = "text-embedding-3-small", batch_size:int=100):
    """Generates embedding payloads from a column in a DataFrame and saves them to a JSONL file."""
    data = df[column_name].to_list()
    batch = []
    with open(filename, "w") as f:
        for x in data:
            batch.append(str(x))
            # Check if batch size is reached; note the off-by-one adjustment
            if len(batch) == batch_size:
                payload = {"model": model_name, "input": batch}
                f.write(json.dumps(payload) + "\n")
                batch = []  # Reset batch
        
        # Handle last batch if it's not empty
        if batch:
            payload = {"model": model_name, "input": batch}
            f.write(json.dumps(payload) + "\n")
    # Read the JSONL file content into a list of dictionaries
    with open(filename, 'r') as f:
        json_list = [json.loads(line) for line in f]

    # Convert the list of dictionaries into a DataFrame
    payloads_df = pl.DataFrame(json_list)

    # Concatenate the original DataFrame column with the generated payloads DataFrame
    result_df = pl.concat([df.select(column_name), payloads_df], how="horizontal")
    
    return result_df

# def generate_embedding_batched_payloads_from_column(filename: str, df: pl.DataFrame, column_name: str, model_name: str = "text-embedding-3-small", batch_size:int=100):
#     """Generates embedding payloads from a column in a DataFrame and saves them to a JSONL file."""
#     data = df[column_name].to_list()
#     batch = []
#     with open(filename, "w") as f:
#         for x in data:
#             # Replace single quotes with double quotes and escape inner double quotes and newline characters
#             x = x.replace("'", '"').replace('"', '\\"').replace('\n', '\\n')
#             batch.append(x)
#             # Check if batch size is reached; note the off-by-one adjustment
#             if len(batch) == batch_size:
#                 payload = {"model": model_name, "input": batch}
#                 f.write(json.dumps(payload) + "\n")
#                 batch = []  # Reset batch

#         # Handle last batch if it's not empty
#         if batch:
#             payload = {"model": model_name, "input": batch}
#             f.write(json.dumps(payload) + "\n")

#     # Read the JSONL file content into a list of dictionaries
#     with open(filename, 'r') as f:
#         json_list = [json.loads(line) for line in f]

#     # Convert the list of dictionaries into a DataFrame
#     payloads_df = pl.DataFrame(json_list)

#     # Concatenate the original DataFrame column with the generated payloads DataFrame
#     result_df = pl.concat([df.select(column_name), payloads_df], how="horizontal")

#     return result_df


def load_openai_emb_results_jsonl(file_path: str, column_name: str) -> pl.DataFrame:
    # Lists to store the extracted data
    models = []
    inputs = []
    embeddings = []

    # Open and read the JSONL file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the JSON content of the current line
            data = json.loads(line)
            
            # Extract and append the model and input text
            models.append(data[0]["model"])
            inputs.append(data[0]["input"])
            
            # For the embeddings, ensure the "data" key exists and extract the embeddings
            if 'data' in data[1] and len(data[1]['data']) > 0:
                embedding = data[1]['data'][0]['embedding']  # Assuming we're interested in the first embedding

                embeddings.append(embedding)
            else:
                embeddings.append([])  # Append an empty list if no embedding is found

    # Create the DataFrame from the lists
    df_out = pl.DataFrame({"model": models, "input": inputs, f"{column_name}_{data[0]["model"]}_embedding": embeddings})
    return df_out

def load_openai_batched_emb_results_jsonl(file_path: str, column_name: str) -> pl.DataFrame:
    # Lists to store the extracted data
    models = []
    inputs = []
    embeddings = []

    # Open and read the JSONL file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the JSON content of the current line
            data = json.loads(line)
            # Check the structure of 'data' to handle both single and batched embeddings
            if isinstance(data[1], dict):
                # Loop through each item in the 'data' list
                for item in data[1]['data']:
                    # Check if the item is an embedding object
                    if item["object"] == "embedding":
                        
                        # Extract and store the embedding
                        embedding = item.get("embedding", [])
                        embeddings.append(embedding)
                for text in data[0]['input']:

                    models.append("text-embedding-3-small")  # Assuming model information is stored at the top level
                    inputs.append(text)  # Assuming input information is stored at the top level
            else:
                # Handle the case for non-standard data formats or missing information
                print(f"Unexpected data format encountered: {data}")
    df_out = pl.DataFrame({
        "model": models,
        "input": inputs,
        f"{column_name}_{data[0]["model"]}_embedding": embeddings
    })
    return df_out

def merge_df_with_openai_emb_results(df:pl.DataFrame,payload_df:pl.DataFrame, openai_results:pl.DataFrame, prompt_column:str) -> pl.DataFrame:
    #first left join the payloads dataframe with the openairesults over the str_messages column and drop the str_messages column
    #then left join the resulting dataframe with the original dataframe over the prompt column
    return df.join(payload_df.join(openai_results,on="input", how="left").select(pl.all().exclude("str_messages")), on=prompt_column, how="left")

def check_max_token_len(df:pl.DataFrame, column_name:str, max_len:int=8192):
    encoding = tiktoken.get_encoding("cl100k_base")
    df = (
        df.
        with_columns(
            pl.col(column_name).map_elements(encoding.encode).alias(f'{column_name}_encoded')
            )
            .with_columns(
                pl.col(f'{column_name}_encoded').map_elements(len).alias('token_len')
            )
        )

    return df.select(pl.col('token_len').max().alias('max_token_len'))[0, 'max_token_len']

def embed_column(df: pl.DataFrame, column_name: str, requests_filepath: str, results_filepath: str, api_key: str, model_name="text-embedding-3-small", batch_size:int=100) -> pl.DataFrame:
    request_url = "https://api.openai.com/v1/embeddings"
    if check_max_token_len(df, column_name) > MAX_INPUT[model_name]:
        raise ValueError(f"Elements in the column exceed the max token length of {model_name}. Max token length is {MAX_INPUT[model_name]}, please remove or truncated the elements that exceed the max len in the column.")
    if model_name not in MAX_INPUT:
        raise ValueError(f"Model name {model_name} is not a valid model name. Please use one of the following model names: {MAX_INPUT.keys()}")
    if column_name not in df.schema:
        raise ValueError(f"Column name {column_name} is not in the dataframe schema. Please use a valid column name.")
    t0 = perf_counter()
    emb_payload_df = generate_embedding_batched_payloads_from_column(requests_filepath, df, column_name, model_name=model_name, batch_size=batch_size)
    print(f'generate_embedding_batched_payloads_from_column took {perf_counter()-t0} minutes/seconds')
    t0 = perf_counter()
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=requests_filepath,
            save_filepath=results_filepath,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(10_000),
            max_tokens_per_minute=float(10_000_000),
            token_encoding_name="cl100k_base",
            max_attempts=int(5),
            logging_level=int(20),
            )
        )
    print(f'process_api_requests_from_file took {perf_counter()-t0} minutes/seconds')
    t0 = perf_counter()
    results_df = load_openai_batched_emb_results_jsonl(results_filepath, column_name=column_name)
    print(f'load_openai_batched_emb_results_jsonl took {perf_counter()-t0} minutes/seconds')
    
    emb_payload_df=emb_payload_df.drop('input').with_columns(pl.col(column_name).alias("input"))
    t0 = perf_counter()
    merged_df = merge_df_with_openai_emb_results(df, emb_payload_df, results_df, column_name)
    print(f'merge_df_with_openai_emb_results took {perf_counter()-t0} minutes/seconds')

    return merged_df.drop(['model_right', "input", "model"])



def embed_columns(
    df: pl.DataFrame, 
    column_names: List[Union[str, List[str]]], 
    models: Union[str, List[str]] = "text-embedding-3-small",
    cache_dir: str = os.path.join(os.path.dirname(os.getcwd()), "cache"),
    api_key: str = "<your_api_key_here>", # Assume API key is passed as a parameter or set elsewhere
) -> pl.DataFrame:
    """
    Modified function to use embed_column for each specified column or merged columns.
    """

    if isinstance(models, str):
        models = [models]  # Ensure models is a list for consistency
    
    for column_set in column_names:
        target_column = column_set
        
        for model_name in models:
            
            # Generate file paths for requests and results
            requests_filepath = os.path.join(cache_dir, f"{target_column}_{model_name}_requests.jsonl")
            results_filepath = os.path.join(cache_dir, f"{target_column}_{model_name}_results.jsonl")
            
            # Generate embeddings and merge them into the DataFrame
            df = embed_column(
                df=df, 
                column_name=target_column, 
                requests_filepath=requests_filepath, 
                results_filepath=results_filepath, 
                api_key=api_key, 
                model_name=model_name
            )
            
            print(f"Embeddings for column '{target_column}' with model '{model_name}' have been merged into the DataFrame.")
    
    return df

def compute_embedding_price(df, text_column, model):
    # Define model prices per token
    model_prices = {
        "text-embedding-3-small": 0.00002 / 1000,
        "text-embedding-3-large": 0.00013 / 1000,
        "ada v2": 0.00010 / 1000,
    }
    
    # Ensure the model is supported
    if model not in model_prices:
        raise ValueError(f"Model {model} not supported. Please choose from {list(model_prices.keys())}.")
    
    # Get the encoding and price for the selected model
    encoding = tiktoken.get_encoding("cl100k_base")
    price_per_token = model_prices[model]
    
    # Compute tokens and their lengths
    df = (
        df
        .with_columns([
        pl.col(text_column).map_elements(encoding.encode).alias('text_encoded'),
            ])
    )
    df = (
        df
        .with_columns([
        pl.col('text_encoded').map_elements(len).alias('token_len')
            ])
    )
    
    # Calculate price for each row
    df = df.with_columns(
        (pl.col('token_len') * price_per_token).alias('price_per_row')
    )
    
    # Compute the total cost
    total_cost = df.select(pl.sum("price_per_row")).to_numpy()[0][0]
    
    return df, total_cost