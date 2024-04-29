from openai import Client
import time
from typing import List,Optional


def get_embedding_single(text:str, model:str="text-embedding-ada-002", client: Optional[Client]=None):
    if client is None:
        client = Client()
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_embedding_list(text_list:List[str], model:str="text-embedding-ada-002", batch_size=100,client: Optional[Client]=None):
    #count time for processing
    if client is None:
        client = Client()
    start = time.time()
    embeddings = []
    if len(text_list) < batch_size:
        print(f"Processing {len(text_list)} chunks of text in a single batch")
        # If the list is smaller than the batch size, process it in one go
        batch_embeddings = client.embeddings.create(input=text_list, model=model).data
        embeddings.extend([item.embedding for item in batch_embeddings])
    else:
        print(f"Processing {len(text_list)} chunks of text in batches")
        # Process the list in batches
        #get number of batches
        for i in range(0, len(text_list), batch_size):
            
            batch = text_list[i:i + batch_size]
            batch_embeddings = client.embeddings.create(input=batch, model=model).data
            embeddings.extend([item.embedding for item in batch_embeddings])
            print(f"Processed  {i} cunks of text out of {len(text_list)}")
    print(f"Embedding Processing took {time.time() - start} seconds")
    return embeddings

