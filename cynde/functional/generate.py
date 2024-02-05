from cynde.async_tools.api_request_parallel_processor import process_api_requests_from_file
from cynde.async_tools.oai_types import ChatCompletion
from cynde.utils.expressions import list_struct_to_string
from typing import List, Union, Optional
import polars as pl
import json
import time
import asyncio
from instructor.function_calls import openai_schema
from pydantic import BaseModel, Field



def generate_chat_completion_payloads(filename:str, data:list[str], prompt:str, model_name="gpt-3.5-turbo-0125"):
    """Hacky in Python with a loop, but it works."""
    with open(filename, "w") as f:
        for x in data:
            # Create a list of messages for each request
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(x)}
            ]

            # Write the messages to the JSONL file
            json_string = json.dumps({"model": model_name, "messages": messages})
            f.write(json_string + "\n")


def generate_chat_completion_with_pydantic_payloads(filename:str, data:list[str], prompt:str,pydantic_model:BaseModel, model_name="gpt-3.5-turbo-0125"):
    """Hacky in Python with a loop, but it works."""
    
    schema = openai_schema(pydantic_model).openai_schema
    print("Using Function Calling",schema["name"])
    tools = [{
                "type": "function",
                "function": schema,
            }]
    
    with open(filename, "w") as f:
        for x in data:
            # Create a list of messages for each request
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(x)}
            ]

            # Write the messages to the JSONL file
            tool_choice={"type": "function", "function": {"name": schema["name"]}}
            json_string = json.dumps({"model": model_name, "messages": messages, "tools": tools, "tool_choice": tool_choice})
            f.write(json_string + "\n")

def generate_chat_payloads_from_column(filename:str, df:pl.DataFrame, column_name:str, prompt:str,pydantic_model:Optional[BaseModel]=None, model_name="gpt-3.5-turbo-0125"):
    """Uses the hacky loop to generate chat payloads from a column in a DataFrame. Has to be rewritten using the Polars Rust backend."""
    data = df[column_name].to_list()
    print("Using Pydantic Model inside",pydantic_model)
    if pydantic_model is None:
        generate_chat_completion_payloads(filename, data, prompt, model_name)
    elif pydantic_model is not None:
        print("Using Pydantic Model")
        generate_chat_completion_with_pydantic_payloads(filename, data, prompt,pydantic_model, model_name)

    #return a dataframe with the generated payloads and the original column name
    #load the generated payloads
    return pl.concat([df.select(pl.col(column_name)), pl.read_ndjson(filename)], how = "horizontal").select(pl.col(column_name),list_struct_to_string("messages"))


def load_openai_results_jsonl(file_path: str) -> pl.DataFrame:
    # Lists to store the extracted data
    messages = []
    choices = []
    usage = []
    results = []

    # Open and read the JSONL file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the JSON content of the current line
            data = json.loads(line)
            
            # Append the required information to the lists
            messages.append(data[0]["messages"])  # First dictionary in the list
            results.append(data[1])  # Second dictionary in the list
            # For list2, ensure the "choices" key exists and has at least one element to extract the "message"
            if 'choices' in data[1] and len(data[1]['choices']) > 0:
                choices.append(data[1]['choices'][0]['message'])
            
            # For list3, simply extract the "usage" dictionary
            if 'usage' in data[1]:
                usage.append(data[1]['usage'])


    df_out = pl.DataFrame({"messages": messages, "choices": choices, "usage": usage, "results": results})
    df_out = df_out.with_columns(list_struct_to_string("messages"))
    return df_out

# Function to load and parse the JSON lines file

def load_openai_results_jsonl_pydantic(file_path: str) -> List[ChatCompletion]:
    completions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            # Assuming the second element in each list is the one to parse
            chat_completion_data = data[1]
            completion = ChatCompletion.model_validate(chat_completion_data)
            completions.append(completion)
    return completions

def merge_df_with_openai_results(df:pl.DataFrame,payload_df:pl.DataFrame, openai_results:pl.DataFrame, prompt_column:str) -> pl.DataFrame:
    #first left join the payloads dataframe with the openairesults over the str_messages column and drop the str_messages column
    #then left join the resulting dataframe with the original dataframe over the prompt column
    return df.join(payload_df.join(openai_results,on="str_messages", how="left").select(pl.all().exclude("str_messages")), on=prompt_column, how="left")

def process_and_merge_llm_responses(df: pl.DataFrame, column_name: str, system_prompt: str, requests_filepath: str, results_filepath: str, api_key: str, pydantic_model:Optional[BaseModel]=None, model_name="gpt-3.5-turbo-0125") -> pl.DataFrame:
    """
    Wrapper function to generate chat payloads from a DataFrame column, process them with LLM, and merge the results back into the DataFrame with timing for each step and overall timing.

    Parameters:
    - df: The source DataFrame.
    - column_name: The name of the column for which to generate prompts.
    - system_prompt: The system prompt to be used for generating chat completions.
    - model_name: The model identifier for the LLM.
    - requests_filepath: File path to save the generated chat completion payloads.
    - results_filepath: File path to save the results from the LLM.
    - api_key: The API key for authentication with the OpenAI API.
    - prompt_column_name: The name for the column containing the processed prompts. Defaults to "processed_prompt".

    Returns:
    - A DataFrame with the results from the LLM merged back.
    """
    # Start the global timer
    global_start_time = time.time()

    # Generate chat payloads from the specified DataFrame column
    print("Generating chat completion payloads...")
    start_time = time.time()
    print("Using Pydantic Model before calling",pydantic_model)
    payload_df = generate_chat_payloads_from_column(requests_filepath, df, column_name, system_prompt,pydantic_model, model_name)
    end_time = time.time()
    print(f"Chat completion payloads generated in {end_time - start_time:.2f} seconds.")

    # Process the chat completion payloads asynchronously
    print("Processing chat completion payloads with the LLM...")
    start_time = time.time()
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=requests_filepath,
            save_filepath=results_filepath,
            request_url="https://api.openai.com/v1/chat/completions",  # or another appropriate endpoint
            api_key=api_key,
            max_requests_per_minute=90000,  # Adjust as necessary
            max_tokens_per_minute=170000,   # Adjust as necessary
            token_encoding_name="cl100k_base",  # Adjust as necessary
            max_attempts=5,
            logging_level=20  # Adjust logging level as necessary
        )
    )
    end_time = time.time()
    print(f"Chat completion payloads processed in {end_time - start_time:.2f} seconds.")

    # Load the results from processing
    print("Loading results from LLM processing...")
    start_time = time.time()
    results_df = load_openai_results_jsonl(results_filepath)
    end_time = time.time()
    print(f"Results loaded in {end_time - start_time:.2f} seconds.")

    # Merge the original DataFrame with the LLM results
    print("Merging LLM results back into the original DataFrame...")
    start_time = time.time()
    merged_df = merge_df_with_openai_results(df, payload_df, results_df, column_name)
    end_time = time.time()
    print(f"LLM results merged in {end_time - start_time:.2f} seconds.")

    # Stop the global timer
    global_end_time = time.time()
    print(f"Total process completed in {global_end_time - global_start_time:.2f} seconds.")

    return merged_df