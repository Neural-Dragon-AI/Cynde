from pydantic import BaseModel

class EmbeddingRequest(BaseModel):
    inputs: str