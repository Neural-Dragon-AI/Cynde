import logfire
logfire.install_auto_tracing(modules=['cynde'])
# logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record='all'))
from typing import List
from pydantic import BaseModel, Field
import polars as pl
# from cynde.functional.generate.modal_gen import generate_column,validate_df
from cynde.functional.embed.embed_oai import embed_column, EmbedConfigOAI, OAIApiFromFileConfig
import modal
import numpy as np

import os
oai_key = os.environ.get("OPENAI_API_KEY")
print(type(oai_key))

texts = ["I saw a puppy a cat and a raccoon during my bike ride in the park","I saw a  crocodile and a snake in the river","DUring camping I saw a bear and a deer"]
df = pl.DataFrame(data={"text":texts})


embed_cfg = EmbedConfigOAI(column_name="text",api_cfg=OAIApiFromFileConfig(api_key=oai_key,requests_filepath="examples/oai_requests.json",save_filepath="examples/oai_results.json"))
print(embed_cfg)
embedded_df = embed_column(df,embed_cfg)
print(embedded_df)