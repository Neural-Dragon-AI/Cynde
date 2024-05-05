import logfire
logfire.install_auto_tracing(modules=['cynde'])
logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record='all'))
from typing import List
from pydantic import BaseModel
import polars as pl
# from cynde.functional.generate.modal_gen import generate_column,validate_df
from cynde.functional.embed.modal_embed import embed_column,EmbedConfig,EmbeddingResponse
import modal
import numpy as np


texts = ["I saw a puppy a cat and a raccoon during my bike ride in the park","I saw a  crocodile and a snake in the river","DUring camping I saw a bear and a deer"]
df = pl.DataFrame(data={"text":texts})
embed_cfg = EmbedConfig(column="text",modal_endpoint="example-tei-bge-small-en-v1.5")

embedded_df = embed_column(df,embed_cfg)
print(embedded_df)