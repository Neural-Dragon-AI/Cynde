import modal
from pydantic import BaseModel,conint,ValidationError
from typing import List, Optional
import pickle
import cloudpickle
import polars as pl
from cynde.functional.generate.types import LLamaInstruction,InstructionConfig
from pydantic._internal._model_construction import ModelMetaclass



def generate_instructions(df:pl.DataFrame, instruction:InstructionConfig) -> List[LLamaInstruction]:
    system_prompt = instruction.system_prompt
    column = instruction.column
    output_schema = instruction.output_schema
    instructions = []
    for text in df[column]:
        instruction = LLamaInstruction(system_prompt=system_prompt, user_message=text, output_schema=output_schema)
        instructions.append(instruction)
    return instructions
    
def generate_column(df:pl.DataFrame, instruction_cfg:InstructionConfig) -> pl.DataFrame:
    f = modal.Function.lookup(instruction_cfg.modal_endpoint, "Model.generate")
    instructions = generate_instructions(df,instruction_cfg)
    requests = []
    for instruction in instructions:
        request = instruction.to_tgi_request()
        requests.append(request)
    responses = []
    responses_generator = f.map(requests)
    # for request in requests:
    #     response = f.remote(request)
    for response in responses_generator:
        responses.append(response.generated_text)
    evaluation = [bool]


    schema_name = instruction_cfg.output_schema["title"]
    input_column_name = instruction_cfg.column
    output_column_name = f"{input_column_name}_{schema_name}"
    df_responses = pl.DataFrame(data={output_column_name:responses})
    df = pl.concat([df,df_responses],how="horizontal")
    return df

def validate_df(df:pl.DataFrame, pydantic_model:BaseModel) -> pl.DataFrame:
    json_schema = pydantic_model.model_json_schema()
    name = json_schema["title"]
    target_cols = [col for col in df.columns if name in col]
    
    for col in target_cols:
        validations = []
        validations_erros = []
        for generation in df[col]:
            try:
                print("generation inside validation:",generation,type(generation))
                validated_model = pydantic_model.model_validate_json(generation)
                validations.append(True)
                validations_erros.append(None)
            except ValidationError as e:
                validations.append(False)
                validations_erros.append(e)
        col_df = pl.DataFrame(data={f"{col}_validations":validations,f"{col}_errors":validations_erros})
        df = pl.concat([df,col_df],how="horizontal")
    return df

