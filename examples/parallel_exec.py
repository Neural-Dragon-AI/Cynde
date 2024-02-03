# run_chat_payload_processing.py
import polars as pl
from typing import List
from datetime import datetime
import cynde.functional as cf
import os
import openai
from cynde.functional.generate import generate_chat_completion_payloads, generate_chat_payloads_from_column
import asyncio
from cynde.async_tools.api_request_parallel_processor import process_api_requests_from_file
from cynde.functional.generate import merge_df_with_openai_results,load_openai_results_jsonl


def generate_demo_df():
    df = pl.DataFrame(
    {
        "customer_id": [101, 102, 103],
        "feedback": [
            "Loved the new product line!",
            "The service was disappointing this time.",
            "Great experience with customer support.",
        ],
        "ratings": [[4, 5, 5], [2, 3, 2], [5, 4, 5]],
        "timestamp": [
            datetime(2023, 1, 1, 14, 30),
            datetime(2023, 1, 2, 9, 15),
            datetime(2023, 1, 3, 18, 45),
        ],
    }
)
    
    return df


if __name__ == "__main__":
    # Set up the OpenAI client with your API key.
    api_key = os.getenv("OPENAI_API_KEY")# 
    print(f"API Key: {api_key}")
    client = openai.Client(api_key=api_key)

    script_dir = os.getcwd()
    # Navigate one directory up to reach /cynde from /cynde/experiments
    parent_dir = os.path.dirname(script_dir)

    # Define the cache directory path as /cynde/cache
    cache_dir = os.path.join(parent_dir, "cache")

    # Ensure the cache directory exists, create if it doesn't
    os.makedirs(cache_dir, exist_ok=True)

    # Define file paths within the /cynde/cache directory
    requests_filepath = os.path.join(cache_dir, "chat_payloads.jsonl")
    results_filepath = os.path.join(cache_dir, "openai_results.jsonl")

    # Generate a demo DataFrame with customer feedback data.
    df = generate_demo_df()
    print("Initial DataFrame:")
    print(df)

    # Generate embeddings for the 'feedback' column using the specified language model.
    print("\nGenerating embeddings for 'feedback' column...")
    embedded_df = cf.embed_columns(df, ["feedback"], client=client)
    print("DataFrame with embeddings:")
    print(embedded_df)

    # Construct prompts dynamically based on the customer feedback data.
    fstring = "Customer ID: {} provided feedback at {} with ratings {} an average rating of {} with a global mean of {}: '{}'"
    system_prompt = "Evaluate the following customer feedback return a True or False based on the sentiment:"
    print("\nGenerating dynamic prompts for each row based on customer feedback...")
    df_prompted = cf.prompt(embedded_df, fstring, [
        pl.col("customer_id"),
        pl.col("timestamp").dt.hour(),  # Convert timestamp to hour
        pl.col("ratings").list.eval(pl.element().cast(pl.Utf8)).list.join("-"),  # Convert list columns to string
        pl.col("ratings").list.mean(),  # Calculate average rating
        pl.col("ratings").list.mean().mean(),  # Calculate global mean rating
        pl.col("feedback")
    ], "customer_prompt")
    print("DataFrame with customer prompts:")
    print(df_prompted)

    # Display the generated prompts for verification.
    print("\nGenerated prompts for each customer feedback:")
    for prompt in df_prompted["customer_prompt"]:
        print(prompt)

    # Prepare chat completion payloads for processing with OpenAI's API.
    print("\nGenerating chat completion payloads...")
    payload_df = generate_chat_payloads_from_column(requests_filepath, df_prompted, "customer_prompt", system_prompt)
    print("Payloads generated and saved to file.")
    print(payload_df)

    # Process API requests to generate chat completions.
    request_url = "https://api.openai.com/v1/chat/completions"  # Adjust as needed for your API endpoint.

    print("\nProcessing API requests for chat completions...")
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=requests_filepath,
            save_filepath=results_filepath,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(90000),
            max_tokens_per_minute=float(170000),
            token_encoding_name="cl100k_base",
            max_attempts=int(5),
            logging_level=int(20),
        )
    )
    print("API requests processed and results saved.")

    # Load the results from the API processing.
    print("\nLoading results from OpenAI processing...")
    results_df = load_openai_results_jsonl(results_filepath)
    print("Results loaded into DataFrame:")
    print(results_df)

    # Merge the original DataFrame with the API results.
    print("\nMerging original DataFrame with API results...")
    merged_df = merge_df_with_openai_results(df_prompted, payload_df, results_df, "customer_prompt")
    print("Merged DataFrame with API results:")
    print(merged_df)


