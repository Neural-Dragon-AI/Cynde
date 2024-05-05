import modal
from pydantic import BaseModel,conint,ValidationError,Field
from typing import List, Optional
import polars as pl
from cynde.functional.embed.types import EmbeddingRequest
import numpy as np

class EmbedConfig(BaseModel):
    column: str
    modal_endpoint: str = Field("example-tei-bge-small-en-v1.5",description="The modal endpoint to use for generating instructions")

class EmbeddingResponse(BaseModel):
    request: EmbeddingRequest
    response: np.ndarray
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

def embed_column(df:pl.DataFrame, embed_cfg: EmbedConfig) -> pl.DataFrame:
    f = modal.Function.lookup(embed_cfg.modal_endpoint, "Model.embed")
    requests = []
    for text in df[embed_cfg.column]:
        request = EmbeddingRequest(inputs=text)
        requests.append(request)
    responses = []
    responses_generator = f.map(requests)
    for response in responses_generator:
        validated_response = EmbeddingResponse(request=request,response=response)
        responses.append(response)
    #vstack the responses
    responses = np.vstack(responses)
    input_column_name = embed_cfg.column
    output_column_name = f"{input_column_name}_{embed_cfg.modal_endpoint}"
    df_responses = pl.DataFrame(data={output_column_name:responses})
    df = pl.concat([df,df_responses],how="horizontal")
    return df

def validate_column(df:pl.DataFrame, embed_cfg: EmbedConfig):
    input_column_name = embed_cfg.column
    output_column_name = f"{input_column_name}_{embed_cfg.modal_endpoint}"
    if output_column_name not in df.columns:
        raise ValueError(f"Column {output_column_name} not found in DataFrame")
    return df

