import polars as pl
import requests

def load_hf_dataset_to_polars_df(author: str, repo: str) -> pl.DataFrame:
    """
    Load a Hugging Face dataset into a Polars DataFrame given an author and dataset name.
    
    Args:
        author (str): The author of the dataset.
        repo (str): The repository name of the dataset.

    Returns:
        pl.DataFrame: The loaded dataset as a Polars DataFrame.
    """
    # Construct the dataset identifier
    dataset_identifier = f"{author}/{repo}"
    
    # Construct the URL to query the dataset viewer API for Parquet files
    dataset_url = f"https://datasets-server.huggingface.co/parquet?dataset={dataset_identifier}"

    # Get the Parquet file URLs from the dataset viewer API
    response = requests.get(dataset_url)
    response.raise_for_status()
    parquet_info = response.json()

    # Extract the Parquet URLs for the "train" split (or any specific split required)
    urls = [f['url'] for f in parquet_info['parquet_files'] if f['split'] == 'train']

    # Read and concatenate all Parquet files into a single Polars DataFrame
    df = pl.concat([pl.read_parquet(url) for url in urls])

    return df

# Example usage
if __name__ == "__main__":
    author = "NeuroDragon"
    repo = "BuggedPythonLeetCode"
    df = load_hf_dataset_to_polars_df(author, repo)
    print(df)
