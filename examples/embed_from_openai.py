import logfire
logfire.install_auto_tracing(modules=['cynde'])
# logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record='all'))
from typing import List
from pydantic import BaseModel, Field
import polars as pl
# from cynde.functional.generate.modal_gen import generate_column,validate_df
from cynde.functional.embed.embed_oai import embed_column, EmbedConfigOAI
import modal
import numpy as np

import os

texts = ["I saw a puppy a cat and a raccoon during my bike ride in the park","I saw a  crocodile and a snake in the river","DUring camping I saw a bear and a deer"]
df = pl.DataFrame(data={"text":texts})





embed_cfg = EmbedConfigOAI(column_name="text",requests_filepath="requests.json",results_filepath="results.json", api_key = os.environ["OPENAI_API_KEY"])

embedded_df = embed_column(df,embed_cfg)
print(embedded_df)