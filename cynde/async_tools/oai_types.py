from typing import List, Optional
from pydantic import BaseModel
from typing_extensions import Literal
from openai.types.chat import ChatCompletion as OriginalChatCompletion, ChatCompletionMessage, ChatCompletionTokenLogprob
from openai.types.chat.chat_completion import ChoiceLogprobs
from openai.types.chat.chat_completion import Choice as OriginalChoice
from openai.types import CompletionUsage

class Choice(OriginalChoice):
    logprobs: Optional[ChoiceLogprobs] = None
    message: Optional[ChatCompletionMessage] = None

class ChatCompletion(OriginalChatCompletion):
    choices: List[Choice]
    system_fingerprint: Optional[str] = None
    """Making system_fingerprint optional to handle cases where it might be null."""
    usage: Optional[CompletionUsage] = None


