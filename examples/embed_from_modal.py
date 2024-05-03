# import logfire
# logfire.install_auto_tracing(modules=['cynde'])
# logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record='all'))
from typing import List
from pydantic import BaseModel
import polars as pl
# from cynde.functional.generate.modal_gen import generate_column,validate_df
from cynde.functional.generate.types import InstructionConfig
import modal
from cynde.deploy.tei import EmbeddingRequest



f = modal.Function.lookup("example-tei-bge-small-en-v1.5", "Model.embed")
texts = ["I saw a puppy a cat and a raccoon during my bike ride in the park","I saw a  crocodile and a snake in the river","DUring camping I saw a bear and a deer"]
for text in texts:
    request = EmbeddingRequest(inputs=text)
    response = f.remote(request)
    print(response)