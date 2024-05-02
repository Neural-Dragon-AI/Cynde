from pydantic import BaseModel, Field
from typing import List, Optional
from cynde.deploy.types import TGIRequest, LLamaInst3Request, Grammar

class LLamaInstruction(BaseModel):
    system_prompt: str
    user_message: str
    output_schema: Optional[dict] = None

    def template(self) -> str:
        system_prompt = self.system_prompt
        user_message = self.user_message
        formatted_prompt =  """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        return formatted_prompt.format(system_prompt=system_prompt,user_message=user_message)
    
    def to_tgi_request(self,request_config: Optional[LLamaInst3Request] = None) -> LLamaInst3Request:
        if request_config is None:
            return LLamaInst3Request(prompt=self.template(),grammar=Grammar(type="json",value=self.output_schema))
        request_config.prompt = self.template()
        return LLamaInst3Request.model_validate(request_config,grammar = Grammar(type="json",value=self.output_schema))

class InstructionConfig(BaseModel):
    system_prompt: str
    column: str
    output_schema: Optional[dict] = None
    modal_endpoint: str = Field("example-tgi-Meta-Llama-3-8B-Instruct",description="The modal endpoint to use for generating instructions")

