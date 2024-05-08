import logfire
logfire.install_auto_tracing(modules=['cynde'])
logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record='all'))
from typing import List
from pydantic import BaseModel
import polars as pl
from cynde.functional.generate.generate_modal import generate_column,validate_df
from cynde.functional.generate.types import InstructionConfig

class Animals(BaseModel):
    location: str
    activity: str
    animals_seen: int  # Constrained integer type
    animals: List[str]

system = "Respond with a valid json parsing the following sentence: \n"
texts = ["I saw a puppy a cat and a raccoon during my bike ride in the park","I saw a  crocodile and a snake in the river","DUring camping I saw a bear and a deer"]
df = pl.DataFrame(data={"text":texts})
instruction = InstructionConfig(system_prompt=system,column="text",output_schema=Animals.model_json_schema(),modal_endpoint="example-tgi-Meta-Llama-3-8B-Instruct")
out_df = generate_column(df,instruction)
validated_df = validate_df(out_df,Animals)

print(validated_df)
for generation,error in zip(validated_df["text_Animals"],validated_df["text_Animals_errors"]):
    print(generation)
    print(error)