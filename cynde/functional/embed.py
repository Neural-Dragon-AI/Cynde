import time
import os
from typing import List, Union, Any, Optional
import polars as pl
from cynde.models.embedders import  get_embedding_list
from openai import Client
import json
import asyncio
from cynde.async_tools.api_request_parallel_processor import process_api_requests_from_file


def generate_embedding_payloads_from_column(filename: str, df: pl.DataFrame, column_name: str, model_name: str = "text-embedding-ada-002"):
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

def merge_df_with_openai_emb_results(df:pl.DataFrame,payload_df:pl.DataFrame, openai_results:pl.DataFrame, prompt_column:str) -> pl.DataFrame:
    #first left join the payloads dataframe with the openairesults over the str_messages column and drop the str_messages column
    #then left join the resulting dataframe with the original dataframe over the prompt column
    return df.join(payload_df.join(openai_results,on="input", how="left").select(pl.all().exclude("str_messages")), on=prompt_column, how="left")

def embed_column(df: pl.DataFrame, column_name: str, requests_filepath: str, results_filepath: str, api_key: str, model_name="text-embedding-3-small") -> pl.DataFrame:
    request_url = "https://api.openai.com/v1/embeddings"
    emb_payload_df = generate_embedding_payloads_from_column(requests_filepath, df, column_name, model_name=model_name)
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=requests_filepath,
            save_filepath=results_filepath,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(90000),
            max_tokens_per_minute=float(10_000_000),
            token_encoding_name="cl100k_base",
            max_attempts=int(5),
            logging_level=int(20),
            )
        )
    results_df = load_openai_emb_results_jsonl(results_filepath, column_name=column_name)
    merged_df = merge_df_with_openai_emb_results(df, emb_payload_df, results_df, column_name)

    return merged_df.drop(['model_right', "input", "model"])



def embed_columns(
    df: pl.DataFrame, 
    column_names: List[Union[str, List[str]]], 
    models: Union[str, List[str]] = "text-embedding-3-small",
    cache_dir: str = os.path.join(os.path.dirname(os.getcwd()), "cache"),
    separator: str = " ", 
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
