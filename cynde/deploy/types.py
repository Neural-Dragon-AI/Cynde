from pydantic import BaseModel, conint, ValidationError, Field
from typing import List, Optional
from text_generation.types import Grammar,Response

class EmbeddingRequest(BaseModel):
    inputs: str
class TGIRequest(BaseModel):
    prompt: str
    do_sample: bool = False
    max_new_tokens: int = 1024
    best_of: Optional[int] = None
    repetition_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    return_full_text: bool = False
    seed: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    truncate: Optional[int] = None
    typical_p: Optional[float] = None
    watermark: bool = False
    decoder_input_details: bool = False
    top_n_tokens: Optional[int] = None
    grammar: Optional[Grammar] = None

class LLamaInst3Request(TGIRequest):
    stop_sequences: Optional[List[str]] =Field(["<|eot_id|>"],description="The stop sequences for LLAMA3 Instruction Tuned")
