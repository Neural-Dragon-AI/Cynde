# Combined Text Dir from cynde

- Full filepath to the merged directory: `C:\Users\Tommaso\Documents\Dev\Cynde\cynde`

- Created: `2024-05-05T12:40:41.963491`

## init

# This is the __init__.py file for the package.


---

## init

# This is the __init__.py file for the package.


---

## api request parallel processor

"""
This script is adapted from work found at:
https://github.com/tiny-rawr/parallel_process_gpt,
which itself is based on examples provided by OpenAI at:
https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py

MIT License

Copyright (c) 2023 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
):
    """
    Asynchronously processes API requests from a given file, executing them in parallel
    while adhering to specified rate limits for requests and tokens per minute.
    
    This function reads a file containing JSONL-formatted API requests, sends these requests
    concurrently to the specified API endpoint, and handles retries for failed attempts,
    all the while ensuring that the execution does not exceed the given rate limits.
    
    Parameters:
    - requests_filepath: Path to the file containing the JSONL-formatted API requests.
    - save_filepath: Path to the file where results or logs should be saved.
    - request_url: The API endpoint URL to which the requests will be sent.
    - api_key: The API key for authenticating requests to the endpoint.
    - max_requests_per_minute: The maximum number of requests allowed per minute.
    - max_tokens_per_minute: The maximum number of tokens (for rate-limited APIs) that can be used per minute.
    - token_encoding_name: Name of the token encoding scheme used for calculating request sizes.
    - max_attempts: The maximum number of attempts for each request in case of failures.
    - logging_level: The logging level to use for reporting the process's progress and issues.
    
    The function initializes necessary tracking structures, sets up asynchronous HTTP sessions,
    and manages request retries and rate limiting. It logs the progress and any issues encountered
    during the process to facilitate monitoring and debugging.
    """
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}
    # use api-key header for Azure deployments
    if '/deployments' in request_url:
        request_header = {"api-key": f"{api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 1, 2, 3, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug(f"File opened. Entering main loop")
        async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(
                            f"Retrying request {next_request.task_id}: {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            # get new request
                            request_json = json.loads(next(requests))
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Reading request {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logging.debug("Read file exhausted")
                            file_not_finished = False

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                # if enough capacity available, call API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (
                        available_request_capacity >= 1
                        and available_token_capacity >= next_request_tokens
                    ):
                        # update counters
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # call API
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None  # reset next_request to empty

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = (
                    time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < seconds_to_pause_after_rate_limit_error
                ):
                    remaining_seconds_to_pause = (
                        seconds_to_pause_after_rate_limit_error
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    logging.warn(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        # after finishing, log final status
        logging.info(
            f"""Parallel processing complete. Results saved to {save_filepath}"""
        )
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )


# dataclasses


@dataclass
class StatusTracker:
    """
    A data class that tracks the progress and status of API request processing.
    
    This class is designed to hold counters for various outcomes of API requests
    (such as successes, failures, and specific types of errors) and other relevant
    metadata to manage and monitor the execution flow of the script.
    
    Attributes:
    - num_tasks_started: The total number of tasks that have been started.
    - num_tasks_in_progress: The current number of tasks that are in progress.
      The script continues running as long as this number is greater than 0.
    - num_tasks_succeeded: The total number of tasks that have completed successfully.
    - num_tasks_failed: The total number of tasks that have failed.
    - num_rate_limit_errors: The count of errors received due to hitting the API's rate limits.
    - num_api_errors: The count of API-related errors excluding rate limit errors.
    - num_other_errors: The count of errors that are neither API errors nor rate limit errors.
    - time_of_last_rate_limit_error: A timestamp (as an integer) of the last time a rate limit error was encountered,
      used to implement a cooling-off period before making subsequent requests.
    
    The class is initialized with all counters set to 0, and the `time_of_last_rate_limit_error`
    set to 0 indicating no rate limit errors have occurred yet.
    """

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """
    Represents an individual API request with associated metadata and the capability to asynchronously call an API.
    
    Attributes:
    - task_id (int): A unique identifier for the task.
    - request_json (dict): The JSON payload to be sent with the request.
    - token_consumption (int): Estimated number of tokens consumed by the request, used for rate limiting.
    - attempts_left (int): The number of retries left if the request fails.
    - metadata (dict): Additional metadata associated with the request.
    - result (list): A list to store the results or errors from the API call.
    
    This class encapsulates the data and actions related to making an API request, including
    retry logic and error handling.
    """

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """
        Asynchronously sends the API request using aiohttp, handles errors, and manages retries.
        
        Parameters:
        - session (aiohttp.ClientSession): The session object used for HTTP requests.
        - request_url (str): The URL to which the request is sent.
        - request_header (dict): Headers for the request, including authorization.
        - retry_queue (asyncio.Queue): A queue for requests that need to be retried.
        - save_filepath (str): The file path where results or errors should be logged.
        - status_tracker (StatusTracker): A shared object for tracking the status of all API requests.
        
        This method attempts to post the request to the given URL. If the request encounters an error,
        it determines whether to retry based on the remaining attempts and updates the status tracker
        accordingly. Successful requests or final failures are logged to the specified file.
        """
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


# functions


def api_endpoint_from_url(request_url: str) -> str:
    """
    Extracts the API endpoint from a given request URL.

    This function applies a regular expression search to find the API endpoint pattern within the provided URL.
    It supports extracting endpoints from standard OpenAI API URLs as well as custom Azure OpenAI deployment URLs.

    Parameters:
    - request_url (str): The full URL of the API request.

    Returns:
    - str: The extracted endpoint from the URL. If the URL does not match expected patterns,
      this function may raise an IndexError for accessing a non-existing match group.

    Example:
    - Input: "https://api.openai.com/v1/completions"
      Output: "completions"
    - Input: "https://custom.azurewebsites.net/openai/deployments/my-model/completions"
      Output: "completions"
    """
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url)
    return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """
    Appends a given JSON payload to the end of a JSON Lines (.jsonl) file.

    Parameters:
    - data: The JSON-serializable Python object (e.g., dict, list) to be appended.
    - filename (str): The path to the .jsonl file to which the data will be appended.

    The function converts `data` into a JSON string and appends it to the specified file,
    ensuring that each entry is on a new line, consistent with the JSON Lines format.

    Note:
    - If the specified file does not exist, it will be created.
    - This function does not return any value.
    """
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request'
                )
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )


def task_id_generator_function():
    """
    Generates a sequence of integer task IDs, starting from 0 and incrementing by 1 each time.

    Yields:
    - An integer representing the next task ID in the sequence.

    This generator function is useful for assigning unique identifiers to tasks or requests
    in a sequence, ensuring each has a distinct ID for tracking and reference purposes.
    """
    task_id = 0
    while True:
        yield task_id
        task_id += 1


# run script


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests_filepath")
    parser.add_argument("--save_filepath", default=None)
    parser.add_argument("--request_url", default="https://api.openai.com/v1/embeddings")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--max_requests_per_minute", type=int, default=3_000 * 0.5)
    parser.add_argument("--max_tokens_per_minute", type=int, default=250_000 * 0.5)
    parser.add_argument("--token_encoding_name", default="cl100k_base")
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--logging_level", default=logging.INFO)
    args = parser.parse_args()

    if args.save_filepath is None:
        args.save_filepath = args.requests_filepath.replace(".jsonl", "_results.jsonl")

    # run script
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=args.requests_filepath,
            save_filepath=args.save_filepath,
            request_url=args.request_url,
            api_key=args.api_key,
            max_requests_per_minute=float(args.max_requests_per_minute),
            max_tokens_per_minute=float(args.max_tokens_per_minute),
            token_encoding_name=args.token_encoding_name,
            max_attempts=int(args.max_attempts),
            logging_level=int(args.logging_level),
        )
    )

---

## oai types

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




---

## init

# This is the __init__.py file for the package.


---

## tei

import os
import subprocess
from pathlib import Path
from pydantic import BaseModel
from typing import List
from modal import App, Image, Mount, Secret, asgi_app, enter, exit, gpu, method
from cynde.functional.embed.types import EmbeddingRequest

MODEL_ID = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 512

LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
    "--max-client-batch-size",
    str(BATCH_SIZE),
    "--max-batch-tokens",
    str(BATCH_SIZE * 512),
]



def download_model():
    subprocess.run(
        [
            "text-embeddings-router",
            "download-weights",
            MODEL_ID,
        ],
    )

app = App("example-tei-" + MODEL_ID.split("/")[-1])
print("App name:", app.name)
latest_version = "ghcr.io/huggingface/text-embeddings-inference:1.2"
tei_image = (
    Image.from_registry(latest_version, add_python="3.12")
    .dockerfile_commands("ENTRYPOINT []")
    
    .run_function(
        download_model,
        secrets=[Secret.from_name("huggingface-secret")],
        timeout=3600,
    )
    .pip_install("httpx")
    .pip_install("numpy")
    .run_commands("pip install pydantic --upgrade")
    .apt_install("git")
    .pip_install("polars","scikit-learn","openai","tiktoken")#, force_build=True)
    
    .run_commands("git clone https://github.com/Neural-Dragon-AI/Cynde/")
    .env({"CYNDE_DIR": "/opt/cynde"})
    .run_commands("cd Cynde && pip install -r requirements.txt && pip install .")
    
    
)

GPU_CONFIG = gpu.A100(count=1,size="40GB")  
@app.cls(
    secrets=[Secret.from_name("huggingface-secret")],
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=15,
    container_idle_timeout=60 * 10,
    timeout=60 * 60,
    image=tei_image,
)
class Model:
    @enter()
    def start_server(self):
        import socket
        import time
        from httpx import AsyncClient

        self.launcher = subprocess.Popen(
            ["text-embeddings-router"] + LAUNCH_FLAGS,
            env={
                **os.environ,
                "HUGGING_FACE_HUB_TOKEN": os.environ["HF_TOKEN"],
            },
        )
        self.client = AsyncClient(base_url="http://127.0.0.1:8000", timeout=60)

        # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
        def webserver_ready():
            try:
                socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
                return True
            except (socket.timeout, ConnectionRefusedError):
                # Check if launcher webserving process has exited.
                # If so, a connection can never be made.
                retcode = self.launcher.poll()
                if retcode is not None:
                    raise RuntimeError(
                        f"launcher exited unexpectedly with code {retcode}"
                    )
                return False

        while not webserver_ready():
            time.sleep(1.0)

        print("Webserver ready!")

    @exit()
    def terminate_server(self):
        self.launcher.terminate()

    @method()
    async def embed(self, request: EmbeddingRequest):
        import numpy as np
        import httpx

        try:
            response = await self.client.post("/embed", json=request.dict())
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
            embeddings = np.array(response.json())
            return embeddings
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e}")
            # Handle the error appropriately, e.g., return an error response
        except Exception as e:
            print(f"An error occurred: {e}")
            # Handle the error appropriately, e.g., return an error response


@app.local_entrypoint()
def main():
    texts = ["I saw a puppy a cat and a raccoon during my bike ride in the park","I saw a  crocodile and a snake in the river","DUring camping I saw a bear and a deer"]
    for text in texts:
        request = EmbeddingRequest(inputs=text)
        response = Model().embed.remote(request)
        print(response)


---

## tgi

import os
import subprocess
from pathlib import Path
from pydantic import BaseModel, conint, ValidationError
from typing import List, Optional
from modal import App, Image, Mount, Secret, asgi_app, enter, exit, gpu, method
from cynde.deploy.types import TGIRequest, LLamaInst3Request

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
   
]

class Prompt(BaseModel):
    system_prompt: str
    user_message: str
    output_schema: Optional[dict] = None
    repetition_penalty: Optional[float] = None

def download_model():
    subprocess.run(
        [
            "text-generation-server",
            "download-weights",
            MODEL_ID,
           
        ],
    )


app = App(
    "example-tgi-" + MODEL_ID.split("/")[-1]
)  # Note: prior to April 2024, "app" was called "stub"
print("App name:", app.name)
latest_version = "ghcr.io/huggingface/text-generation-inference:sha-f9cf345"
version_from_example = "ghcr.io/huggingface/text-generation-inference:1.4"
tgi_image = (
    Image.from_registry(latest_version)
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(
        download_model,
        secrets=[Secret.from_name("huggingface-secret")],
        timeout=3600,
    )
    .run_commands("pip install pydantic --upgrade")
    .pip_install("outlines")
    .pip_install("text-generation")
)

GPU_CONFIG = gpu.A100(count=1,size="40GB")  # 2 H100s


@app.cls(
    secrets=[Secret.from_name("huggingface-secret")],
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=15,
    container_idle_timeout=60 * 10,
    timeout=60 * 60,
    image=tgi_image,
)
class Model:
    @enter()
    def start_server(self):
        import socket
        import time

        from text_generation import AsyncClient

        self.launcher = subprocess.Popen(
            ["text-generation-launcher"] + LAUNCH_FLAGS,
            env={
                **os.environ,
                "HUGGING_FACE_HUB_TOKEN": os.environ["HF_TOKEN"],
            },
        )
        self.client = AsyncClient("http://127.0.0.1:8000", timeout=60)
        
        
        self.template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

        # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
        def webserver_ready():
            try:
                socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
                return True
            except (socket.timeout, ConnectionRefusedError):
                # Check if launcher webserving process has exited.
                # If so, a connection can never be made.
                retcode = self.launcher.poll()
                if retcode is not None:
                    raise RuntimeError(
                        f"launcher exited unexpectedly with code {retcode}"
                    )
                return False

        while not webserver_ready():
            time.sleep(1.0)

        print("Webserver ready!")

    @exit()
    def terminate_server(self):
        self.launcher.terminate()

    @method()
    async def generate(self, request: LLamaInst3Request) :  
        result = await self.client.generate(**request.model_dump())

        return result


---

## types

from pydantic import BaseModel, conint, ValidationError, Field
from typing import List, Optional
from text_generation.types import Grammar,Response


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


---

## init

# This is the __init__.py file for the package.
from .embed import *
from .prompt import *
from .cv import *
from .classify import *
from .generate import *
from .results import *


from dotenv import load_dotenv

load_dotenv()


def set_directories(root_dir):
    if root_dir is None:
        raise ValueError("CYNDE_DIR environment variable must be set before using cynde")
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(os.path.join(root_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "output"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "cynde_mount"), exist_ok=True)
    os.environ['CACHE_DIR'] = os.path.join(root_dir, "cache")
    os.environ['OUTPUT_DIR'] = os.path.join(root_dir, "output")
    os.environ['MODAL_MOUNT'] = os.path.join(root_dir, "cynde_mount")

root_dir = os.getenv('CYNDE_DIR')
print(root_dir)
set_directories(root_dir)

---

## init

# This is the __init__.py file for the package.


---

## embed

import time
from time import perf_counter
import os
from typing import List, Union, Any, Optional
import polars as pl
from openai import Client
import json
import tiktoken

import asyncio
from cynde.async_tools.api_request_parallel_processor import process_api_requests_from_file

MAX_INPUT = {
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191,
    "togethercomputer/m2-bert-80M-32k-retrieval": 32_000,
    "voyage-code-2":16_000
}


def generate_embedding_payloads_from_column(filename: str, df: pl.DataFrame, column_name: str, model_name: str = "text-embedding-3-small"):
    """Generates embedding payloads from a column in a DataFrame and saves them to a JSONL file."""
    # Extract the data from the specified column as a list
    data = df[column_name].to_list()
    
    with open(filename, "w") as f:
        for x in data:
            # Create the payload for each text entry
            payload = {"model": model_name, "input": str(x)}
            
            # Write the payload to the JSONL file
            json_string = json.dumps(payload)
            f.write(json_string + "\n")

    # Read the JSONL file content into a list of dictionaries
    with open(filename, 'r') as f:
        json_list = [json.loads(line) for line in f]

    # Convert the list of dictionaries into a DataFrame
    payloads_df = pl.DataFrame(json_list)

    # Concatenate the original DataFrame column with the generated payloads DataFrame
    result_df = pl.concat([df.select(column_name), payloads_df], how="horizontal")
    
    return result_df

def generate_embedding_batched_payloads_from_column(filename: str, df: pl.DataFrame, column_name: str, model_name: str = "text-embedding-3-small", batch_size:int=100):
    """Generates embedding payloads from a column in a DataFrame and saves them to a JSONL file."""
    data = df[column_name].to_list()
    batch = []
    with open(filename, "w") as f:
        for x in data:
            batch.append(str(x))
            # Check if batch size is reached; note the off-by-one adjustment
            if len(batch) == batch_size:
                payload = {"model": model_name, "input": batch}
                f.write(json.dumps(payload) + "\n")
                batch = []  # Reset batch
        
        # Handle last batch if it's not empty
        if batch:
            payload = {"model": model_name, "input": batch}
            f.write(json.dumps(payload) + "\n")
    # Read the JSONL file content into a list of dictionaries
    with open(filename, 'r') as f:
        json_list = [json.loads(line) for line in f]

    # Convert the list of dictionaries into a DataFrame
    payloads_df = pl.DataFrame(json_list)

    # Concatenate the original DataFrame column with the generated payloads DataFrame
    result_df = pl.concat([df.select(column_name), payloads_df], how="horizontal")
    
    return result_df



def load_openai_emb_results_jsonl(file_path: str, column_name: str) -> pl.DataFrame:
    # Lists to store the extracted data
    models = []
    inputs = []
    embeddings = []

    # Open and read the JSONL file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the JSON content of the current line
            data = json.loads(line)
            
            # Extract and append the model and input text
            models.append(data[0]["model"])
            inputs.append(data[0]["input"])
            
            # For the embeddings, ensure the "data" key exists and extract the embeddings
            if 'data' in data[1] and len(data[1]['data']) > 0:
                embedding = data[1]['data'][0]['embedding']  # Assuming we're interested in the first embedding

                embeddings.append(embedding)
            else:
                embeddings.append([])  # Append an empty list if no embedding is found

    # Create the DataFrame from the lists
    model_value = data[0]["model"]
    df_out = pl.DataFrame({"model": models, "input": inputs, f"{column_name}_{model_value}_embedding": embeddings})

    return df_out

def load_openai_batched_emb_results_jsonl(file_path: str, column_name: str) -> pl.DataFrame:
    # Lists to store the extracted data
    models = []
    inputs = []
    embeddings = []

    # Open and read the JSONL file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the JSON content of the current line
            data = json.loads(line)
            # Check the structure of 'data' to handle both single and batched embeddings
            if isinstance(data[1], dict):
                # Loop through each item in the 'data' list
                for item in data[1]['data']:
                    # Check if the item is an embedding object
                    if item["object"] == "embedding":
                        
                        # Extract and store the embedding
                        embedding = item.get("embedding", [])
                        embeddings.append(embedding)
                for text in data[0]['input']:

                    models.append("text-embedding-3-small")  # Assuming model information is stored at the top level
                    inputs.append(text)  # Assuming input information is stored at the top level
            else:
                # Handle the case for non-standard data formats or missing information
                print(f"Unexpected data format encountered: {data}")
    model_key = data[0]["model"]  # Intermediate variable
    df_out = pl.DataFrame({
        "model": models,
        "input": inputs,
        f"{column_name}_{model_key}_embedding": embeddings  # Using the intermediate variable
    })

    return df_out

def merge_df_with_openai_emb_results(df:pl.DataFrame,payload_df:pl.DataFrame, openai_results:pl.DataFrame, prompt_column:str) -> pl.DataFrame:
    #first left join the payloads dataframe with the openairesults over the str_messages column and drop the str_messages column
    #then left join the resulting dataframe with the original dataframe over the prompt column
    return df.join(payload_df.join(openai_results,on="input", how="left").select(pl.all().exclude("str_messages")), on=prompt_column, how="left")

def check_max_token_len(df:pl.DataFrame, column_name:str, max_len:int=8192):
    encoding = tiktoken.get_encoding("cl100k_base")
    df = (
        df.
        with_columns(
            pl.col(column_name).map_elements(encoding.encode).alias(f'{column_name}_encoded')
            )
            .with_columns(
                pl.col(f'{column_name}_encoded').map_elements(len).alias('token_len')
            )
        )

    return df.select(pl.col('token_len').max().alias('max_token_len'))[0, 'max_token_len']

def embed_column(df: pl.DataFrame, column_name: str, requests_filepath: str, results_filepath: str, api_key: str, model_name="text-embedding-3-small", request_url = "https://api.openai.com/v1/embeddings", batch_size:int=5) -> pl.DataFrame:

    if check_max_token_len(df, column_name) > MAX_INPUT[model_name]:
        raise ValueError(f"Elements in the column exceed the max token length of {model_name}. Max token length is {MAX_INPUT[model_name]}, please remove or truncated the elements that exceed the max len in the column.")
    if model_name not in MAX_INPUT:
        raise ValueError(f"Model name {model_name} is not a valid model name. Please use one of the following model names: {MAX_INPUT.keys()}")
    if column_name not in df.schema:
        raise ValueError(f"Column name {column_name} is not in the dataframe schema. Please use a valid column name.")
    t0 = perf_counter()
    emb_payload_df = generate_embedding_batched_payloads_from_column(requests_filepath, df, column_name, model_name=model_name, batch_size=batch_size)
    print(f'generate_embedding_batched_payloads_from_column took {perf_counter()-t0} minutes/seconds')
    t0 = perf_counter()
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=requests_filepath,
            save_filepath=results_filepath,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(100),#10_000
            max_tokens_per_minute=float(1_000_000),#10_000_000
            token_encoding_name="cl100k_base",
            max_attempts=int(5),
            logging_level=int(20),
            )
        )
    print(f'process_api_requests_from_file took {perf_counter()-t0} minutes/seconds')
    t0 = perf_counter()
    results_df = load_openai_batched_emb_results_jsonl(results_filepath, column_name=column_name)
    print(f'load_openai_batched_emb_results_jsonl took {perf_counter()-t0} minutes/seconds')
    
    emb_payload_df=emb_payload_df.drop('input').with_columns(pl.col(column_name).alias("input"))
    t0 = perf_counter()
    merged_df = merge_df_with_openai_emb_results(df, emb_payload_df, results_df, column_name)
    print(f'merge_df_with_openai_emb_results took {perf_counter()-t0} minutes/seconds')

    return merged_df.drop(['model_right', "input", "model"])



def embed_columns(
    df: pl.DataFrame, 
    column_names: List[Union[str, List[str]]], 
    models: Union[str, List[str]] = "text-embedding-3-small",
    cache_dir: str = os.path.join(os.path.dirname(os.getcwd()), "cache"),
    request_url = "https://api.openai.com/v1/embeddings",
    api_key: str = "<your_api_key_here>", # Assume API key is passed as a parameter or set elsewhere
) -> pl.DataFrame:
    """
    Modified function to use embed_column for each specified column or merged columns.
    """

    if isinstance(models, str):
        models = [models]  # Ensure models is a list for consistency
    
    for column_set in column_names:
        target_column = column_set
        
        for model_name in models:
            
            # Generate file paths for requests and results
            save_model_name = model_name.replace("/", "")
            requests_filepath = os.path.join(os.environ.get('CACHE_DIR'), f"{target_column}_{save_model_name}_requests.jsonl")
            results_filepath = os.path.join(os.environ.get('OUTPUT_DIR'), f"{target_column}_{save_model_name}_results.jsonl")
            if os.path.exists(requests_filepath) or os.path.exists(results_filepath):
                time_code = time.strftime("%Y-%m-%d_%H-%M-%S")
                requests_filepath = os.path.join(os.environ.get('CACHE_DIR'), f"{target_column}_{save_model_name}_{time_code}_requests.jsonl")
                results_filepath = os.path.join(os.environ.get('OUTPUT_DIR'), f"{target_column}_{save_model_name}_{time_code}_results.jsonl")
                
            
            # Generate embeddings and merge them into the DataFrame
            df = embed_column(
                df=df, 
                column_name=target_column, 
                requests_filepath=requests_filepath, 
                results_filepath=results_filepath, 
                api_key=api_key, 
                request_url=request_url,
                model_name=model_name
            )
            
            print(f"Embeddings for column '{target_column}' with model '{model_name}' have been merged into the DataFrame.")
    
    return df

def compute_embedding_price(df:pl.DataFrame, text_column:str, model:str):
    # Define model prices per token
    model_prices = {
        "text-embedding-3-small": 0.00002 / 1000,
        "text-embedding-3-large": 0.00013 / 1000,
        "ada v2": 0.00010 / 1000,
    }
    
    # Ensure the model is supported
    if model not in model_prices:
        raise ValueError(f"Model {model} not supported. Please choose from {list(model_prices.keys())}.")
    
    # Get the encoding and price for the selected model
    encoding = tiktoken.get_encoding("cl100k_base")
    price_per_token = model_prices[model]
    
    # Compute tokens and their lengths
    df = (
        df
        .with_columns([
        pl.col(text_column).map_elements(encoding.encode).alias('text_encoded'),
            ])
    )
    df = (
        df
        .with_columns([
        pl.col('text_encoded').map_elements(len).alias('token_len')
            ])
    )
    
    # Calculate price for each row
    df = df.with_columns(
        (pl.col('token_len') * price_per_token).alias('price_per_row')
    )
    
    # Compute the total cost
    total_cost = df.select(pl.sum("price_per_row")).to_numpy()[0][0]
    
    return df, total_cost

---

## init

# This is the __init__.py file for the package.


---

## modal embed

import modal
from pydantic import BaseModel,conint,ValidationError,Field
from typing import List, Optional
import polars as pl
from cynde.functional.embed.types import EmbeddingRequest
import numpy as np

class EmbedConfig(BaseModel):
    column: str
    modal_endpoint: str = Field("example-tei-bge-small-en-v1.5",description="The modal endpoint to use for generating instructions")

class EmbeddingResponse(BaseModel):
    request: EmbeddingRequest
    response: np.ndarray
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

def embed_column(df:pl.DataFrame, embed_cfg: EmbedConfig) -> pl.DataFrame:
    f = modal.Function.lookup(embed_cfg.modal_endpoint, "Model.embed")
    requests = []
    for text in df[embed_cfg.column]:
        request = EmbeddingRequest(inputs=text)
        requests.append(request)
    responses = []
    responses_generator = f.map(requests)
    for response in responses_generator:
        validated_response = EmbeddingResponse(request=request,response=response)
        responses.append(response)
    #vstack the responses
    responses = np.vstack(responses)
    input_column_name = embed_cfg.column
    output_column_name = f"{input_column_name}_{embed_cfg.modal_endpoint}"
    df_responses = pl.DataFrame(data={output_column_name:responses})
    df = pl.concat([df,df_responses],how="horizontal")
    return df

def validate_column(df:pl.DataFrame, embed_cfg: EmbedConfig):
    input_column_name = embed_cfg.column
    output_column_name = f"{input_column_name}_{embed_cfg.modal_endpoint}"
    if output_column_name not in df.columns:
        raise ValueError(f"Column {output_column_name} not found in DataFrame")
    return df



---

## types

from pydantic import BaseModel

class EmbeddingRequest(BaseModel):
    inputs: str

---

## generate

from cynde.async_tools.api_request_parallel_processor import process_api_requests_from_file
from cynde.async_tools.oai_types import ChatCompletion
from cynde.utils.expressions import list_struct_to_string
from typing import List, Union, Optional
import polars as pl
import json
import os
import time
import asyncio
from instructor.function_calls import openai_schema
from pydantic import BaseModel, Field



def generate_chat_completion_payloads(filename:str, data:list[str], prompt:str, model_name="gpt-3.5-turbo-0125"):
    """Hacky in Python with a loop, but it works."""
    with open(filename, "w") as f:
        for x in data:
            # Create a list of messages for each request
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(x)}
            ]

            # Write the messages to the JSONL file
            json_string = json.dumps({"model": model_name, "messages": messages})
            f.write(json_string + "\n")


def generate_chat_completion_with_pydantic_payloads(filename:str, data:list[str], prompt:str,pydantic_model:BaseModel, model_name="gpt-3.5-turbo-0125"):
    """Hacky in Python with a loop, but it works."""
    
    schema = openai_schema(pydantic_model).openai_schema
    print("Using Function Calling",schema["name"])
    tools = [{
                "type": "function",
                "function": schema,
            }]
    
    with open(filename, "w") as f:
        for x in data:
            # Create a list of messages for each request
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(x)}
            ]

            # Write the messages to the JSONL file
            tool_choice={"type": "function", "function": {"name": schema["name"]}}
            json_string = json.dumps({"model": model_name, "messages": messages, "tools": tools, "tool_choice": tool_choice})
            f.write(json_string + "\n")

def generate_chat_payloads_from_column(filename:str, df:pl.DataFrame, column_name:str, prompt:str,pydantic_model:Optional[BaseModel]=None, model_name="gpt-3.5-turbo-0125"):
    """Uses the hacky loop to generate chat payloads from a column in a DataFrame. Has to be rewritten using the Polars Rust backend."""
    data = df[column_name].to_list()
    print("Using Pydantic Model inside",pydantic_model)
    if pydantic_model is None:
        generate_chat_completion_payloads(filename, data, prompt, model_name)
    elif pydantic_model is not None:
        print("Using Pydantic Model")
        generate_chat_completion_with_pydantic_payloads(filename, data, prompt,pydantic_model, model_name)

    #return a dataframe with the generated payloads and the original column name
    #load the generated payloads
    return pl.concat([df.select(pl.col(column_name)), pl.read_ndjson(filename)], how = "horizontal").select(pl.col(column_name),list_struct_to_string("messages"))


def load_openai_results_jsonl(file_path: str) -> pl.DataFrame:
    # Lists to store the extracted data
    messages = []
    choices = []
    usage = []
    results = []

    # Open and read the JSONL file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the JSON content of the current line
            data = json.loads(line)
            
            # Append the required information to the lists
            messages.append(data[0]["messages"])  # First dictionary in the list
            results.append(data[1])  # Second dictionary in the list
            # For list2, ensure the "choices" key exists and has at least one element to extract the "message"
            if 'choices' in data[1] and len(data[1]['choices']) > 0:
                choices.append(data[1]['choices'][0]['message'])
            
            # For list3, simply extract the "usage" dictionary
            if 'usage' in data[1]:
                usage.append(data[1]['usage'])


    df_out = pl.DataFrame({"messages": messages, "choices": choices, "usage": usage, "results": results})
    df_out = df_out.with_columns(list_struct_to_string("messages"))
    return df_out

# Function to load and parse the JSON lines file

def load_openai_results_jsonl_pydantic(file_path: str) -> List[ChatCompletion]:
    completions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            # Assuming the second element in each list is the one to parse
            chat_completion_data = data[1]
            completion = ChatCompletion.model_validate(chat_completion_data)
            completions.append(completion)
    return completions

def merge_df_with_openai_results(df:pl.DataFrame,payload_df:pl.DataFrame, openai_results:pl.DataFrame, prompt_column:str) -> pl.DataFrame:
    #first left join the payloads dataframe with the openairesults over the str_messages column and drop the str_messages column
    #then left join the resulting dataframe with the original dataframe over the prompt column
    return df.join(payload_df.join(openai_results,on="str_messages", how="left").select(pl.all().exclude("str_messages")), on=prompt_column, how="left")

def process_and_merge_llm_responses(df: pl.DataFrame, column_name: str, system_prompt: str, api_key: str, pydantic_model:Optional[BaseModel]=None, model_name="gpt-3.5-turbo-0125") -> pl.DataFrame:
    """
    Wrapper function to generate chat payloads from a DataFrame column, process them with LLM, and merge the results back into the DataFrame with timing for each step and overall timing.

    Parameters:
    - df: The source DataFrame.
    - column_name: The name of the column for which to generate prompts.
    - system_prompt: The system prompt to be used for generating chat completions.
    - model_name: The model identifier for the LLM.
    - requests_filepath: File path to save the generated chat completion payloads.
    - results_filepath: File path to save the results from the LLM.
    - api_key: The API key for authentication with the OpenAI API.
    - prompt_column_name: The name for the column containing the processed prompts. Defaults to "processed_prompt".

    Returns:
    - A DataFrame with the results from the LLM merged back.
    """
    # Start the global timer
    global_start_time = time.time()
    target_column = column_name
    requests_filepath = os.path.join(os.environ.get('CACHE_DIR'), f"{target_column}_{model_name}_requests.jsonl")
    results_filepath = os.path.join(os.environ.get('OUTPUT_DIR'), f"{target_column}_{model_name}_results.jsonl")
    if os.path.exists(requests_filepath) or os.path.exists(results_filepath):
        time_code = time.strftime("%Y-%m-%d_%H-%M-%S")
        requests_filepath = os.path.join(os.environ.get('CACHE_DIR'), f"{target_column}_{model_name}_{time_code}_requests.jsonl")
        results_filepath = os.path.join(os.environ.get('OUTPUT_DIR'), f"{target_column}_{model_name}_{time_code}_results.jsonl")

    # Generate chat payloads from the specified DataFrame column
    print("Generating chat completion payloads...")
    start_time = time.time()
    print("Using Pydantic Model before calling",pydantic_model)
    payload_df = generate_chat_payloads_from_column(requests_filepath, df, column_name, system_prompt,pydantic_model, model_name)
    end_time = time.time()
    print(f"Chat completion payloads generated in {end_time - start_time:.2f} seconds.")

    
                

    # Process the chat completion payloads asynchronously
    print("Processing chat completion payloads with the LLM...")
    start_time = time.time()
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=requests_filepath,
            save_filepath=results_filepath,
            request_url="https://api.openai.com/v1/chat/completions",  # or another appropriate endpoint
            api_key=api_key,
            max_requests_per_minute=90000,  # Adjust as necessary
            max_tokens_per_minute=170000,   # Adjust as necessary
            token_encoding_name="cl100k_base",  # Adjust as necessary
            max_attempts=5,
            logging_level=20  # Adjust logging level as necessary
        )
    )
    end_time = time.time()
    print(f"Chat completion payloads processed in {end_time - start_time:.2f} seconds.")

    # Load the results from processing
    print("Loading results from LLM processing...")
    start_time = time.time()
    results_df = load_openai_results_jsonl(results_filepath)
    end_time = time.time()
    print(f"Results loaded in {end_time - start_time:.2f} seconds.")

    # Merge the original DataFrame with the LLM results
    print("Merging LLM results back into the original DataFrame...")
    start_time = time.time()
    merged_df = merge_df_with_openai_results(df, payload_df, results_df, column_name)
    end_time = time.time()
    print(f"LLM results merged in {end_time - start_time:.2f} seconds.")

    # Stop the global timer
    global_end_time = time.time()
    print(f"Total process completed in {global_end_time - global_start_time:.2f} seconds.")

    return merged_df

---

## init

# This is the __init__.py file for the package.


---

## modal gen

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
    for request in requests:
        response = f.remote(request)
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



---

## types

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



---

## init

# This is the __init__.py file for the package.


---

## classify

from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import polars as pl
from typing import Tuple
import time
from cynde.functional.predict.types import PipelineResults,PredictConfig,PipelineInput,FeatureSet,InputConfig,ClassifierConfig,BaseClassifierConfig, LogisticRegressionConfig, RandomForestClassifierConfig, HistGradientBoostingClassifierConfig, CVConfig
from cynde.functional.predict.cv import train_test_val,generate_nested_cv
from cynde.functional.predict.preprocess import load_preprocessed_features,check_add_cv_index,validate_preprocessed_inputs

def create_pipeline(df: pl.DataFrame, feature_set: FeatureSet, classifier_config: BaseClassifierConfig) -> Pipeline:
    """ maybne the df.schema is enough and we do not need to pass the whole df """
    transformers = []
    numerical_features = [feature.column_name for feature in feature_set.numerical]
    if numerical_features:
        scaler = feature_set.numerical[0].get_scaler()  # Assuming all numerical features use the same scaler
        transformers.append(("numerical", scaler, numerical_features))
    embedding_features = [feature.column_name for feature in feature_set.embeddings]
    if embedding_features:
        #embedding features are stored as list[float] in polars but we map them to multiple columns of float in sklearn
        # so here we assume that we already pre-processed each embedding_feature to bea  lsit of columns of format column_name_{i}
        #accumulate for each embedding feature the list of columns that represent it and flatten it
        embedding_features = [f"{feature}_{i}" for feature in embedding_features for i in range(0,feature_set.embeddings[0].embedding_size)]
        scaler = feature_set.embeddings[0].get_scaler()  # Assuming all embedding features use the same scaler
        transformers.append(("embedding", scaler, embedding_features))

    categorical_features = [feature.column_name for feature in feature_set.categorical]
    if categorical_features:
        for feature in feature_set.categorical:
            if feature.one_hot_encoding:
                if df[feature.column_name].dtype == pl.Categorical:
                    categories = [df[feature.column_name].unique().to_list()]
                elif df[feature.column_name].dtype == pl.Enum:
                    categories = [df[feature.column_name].dtype.categories]
                else:
                    raise ValueError(f"Column '{feature.column_name}' must be of type pl.Categorical or pl.Enum for one-hot encoding.")
                one_hot_encoder = OneHotEncoder(categories=categories, handle_unknown='error', sparse_output=False)
                transformers.append((f"categorical_{feature.column_name}", one_hot_encoder, [feature.column_name]))
            else:
                if df[feature.column_name].dtype not in [pl.Float32, pl.Float64]:
                    raise ValueError(f"Column '{feature.column_name}' must be of type pl.Float32 or pl.Float64 for physical representation.")
                transformers.append((f"categorical_{feature.column_name}", "passthrough", [feature.column_name]))

    preprocessor = ColumnTransformer(transformers)

    # Create the classifier based on the classifier configuration
    if isinstance(classifier_config, LogisticRegressionConfig):
        classifier = LogisticRegression(**classifier_config.dict(exclude={"classifier_name"}))
    elif isinstance(classifier_config, RandomForestClassifierConfig):
        classifier = RandomForestClassifier(**classifier_config.dict(exclude={"classifier_name"}))
    elif isinstance(classifier_config, HistGradientBoostingClassifierConfig):
        classifier = HistGradientBoostingClassifier(**classifier_config.dict(exclude={"classifier_name"}))
    else:
        raise ValueError(f"Unsupported classifier: {classifier_config.classifier_name}")

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    pipeline.set_output(transform="polars")
    return pipeline

def evaluate_model(pipeline: Pipeline, X, y):
    """ Gotta make sure the returned predictions have the cv_index column"""
    predictions = pipeline.predict(X)
    accuracy = accuracy_score(y, predictions)
    mcc = matthews_corrcoef(y,predictions)
    pred_df = pl.DataFrame({"cv_index":X["cv_index"],"predictions":predictions})
    return pred_df,accuracy, mcc


def predict_pipeline(input_config:InputConfig,pipeline_input:PipelineInput) -> Tuple[pl.DataFrame,pl.DataFrame,float,float]:
    feature_set = input_config.feature_sets[pipeline_input.feature_index]
    df_fold = load_preprocessed_features(input_config,pipeline_input.feature_index)
    print(df_fold)
    df_train,df_val,df_test = train_test_val(df_fold,pipeline_input.train_idx,pipeline_input.val_idx,pipeline_input.test_idx)
    print(df_train)
    pipeline = create_pipeline(df_train, feature_set, pipeline_input.cls_config)
    print(pipeline)
    pipeline.fit(df_train,df_train["target"])
    train_predictions, train_accuracy, train_mcc = evaluate_model(pipeline, df_train, df_train["target"])
    val_predictions,val_accuracy, val_mcc = evaluate_model(pipeline, df_val, df_val["target"])
    test_predictions,test_accuracy,test_mcc = evaluate_model(pipeline, df_test, df_test["target"])
    return PipelineResults(train_predictions = train_predictions,
                           val_predictions=val_predictions,
                           test_predictions=test_predictions,
                           train_accuracy=train_accuracy,
                           train_mcc=train_mcc,
                           val_accuracy=val_accuracy,
                           val_mcc=val_mcc,
                           test_accuracy=test_accuracy,
                           test_mcc=test_mcc)




def train_nested_cv(df:pl.DataFrame, task_config:PredictConfig) -> pl.DataFrame:
    """ Deploy a CV training pipeline to Modal, it requires a df with cv_index column and the features set to have already pre-processed and cached 
    1) Validate the input_config and check if the preprocessed features are present locally 
    2) create a generator that yields the modal path to the features and targets frames as well as the scikit pipeline object 
    3) execute through a modal starmap a script that fit end eval each pipeline on each feature set and return the results
    4) collect and aggregate the results locally and save and return the results
    """
    #validate the inputs and check if the preprocessed features are present locally
    df = check_add_cv_index(df,strict=True)
    validate_preprocessed_inputs(task_config.input_config)
    
    #extract the subset of columns necessary for constructing the cross validation folds 
    unique_groups = list(set(task_config.cv_config.inner.groups + task_config.cv_config.outer.groups))
    df_idx = df.select(pl.col("cv_index"),pl.col(unique_groups))

    nested_cv = generate_nested_cv(df_idx,task_config)

    for pipeline_input in nested_cv:
        start = time.time()
        print(f"Training pipeline with classifier {pipeline_input.cls_config.classifier_name} on feature set {task_config.input_config.feature_sets[pipeline_input.feature_index]}")
        results = predict_pipeline(task_config.input_config,pipeline_input)
        print(results)
        end = time.time()
        print(f"Training pipeline took {end-start} seconds")


---

## cv

import polars as pl
from pydantic import BaseModel
from typing import List, Optional, Tuple, Generator
import itertools
from cynde.functional.predict.types import PredictConfig, BaseFoldConfig,PipelineInput,BaseClassifierConfig,ClassifierConfig,InputConfig,CVConfig, KFoldConfig, PurgedConfig, StratifiedConfig, CVSummary

from cynde.functional.predict.preprocess import check_add_cv_index




def shuffle_frame(df:pl.DataFrame):
    return df.sample(fraction=1,shuffle=True)

def slice_frame(df:pl.DataFrame, num_slices:int, shuffle:bool = False, explode:bool = False) -> List[pl.DataFrame]:
    max_index = df.shape[0]
    if shuffle:
        df = shuffle_frame(df)
    indexes = [0] + [max_index//num_slices*i for i in range(1,num_slices)] + [max_index]
    if explode:
        return [df.slice(indexes[i],indexes[i+1]-indexes[i]).explode("cv_index").select(pl.col("cv_index")) for i in range(len(indexes)-1)]
    else:
        return [df.slice(indexes[i],indexes[i+1]-indexes[i]).select(pl.col("cv_index")) for i in range(len(indexes)-1)]

def hacky_list_relative_slice(list: List[int], k: int):
    slices = {}
    slice_size = len(list) // k
    for i in range(k):
        if i < k - 1:
            slices["fold_{}".format(i)] = list[i*slice_size:(i+1)*slice_size]
        else:
            # For the last slice, include the remainder
            slices["fold_{}".format(i)] = list[i*slice_size:]
    return slices

def kfold_combinatorial(df: pl.DataFrame, config: KFoldConfig) -> CVSummary:
    df = check_add_cv_index(df,strict=True)
    cv_index = df["cv_index"].shuffle(seed=config.random_state)
    num_samples = cv_index.shape[0]
    fold_size = num_samples // config.k
    index_start = pl.Series([int(i*fold_size) for i in range(config.k)])
    train_indexes = []
    test_indexes = []
    fold_numbers = []
    
    print("index_start",index_start)
    folds = [cv_index.slice(offset= start,length=fold_size) for start in index_start]
    #use iter-tools to compute all combinations of indexes in train ant test let's assume only combinatorial for now
    # folds are indexed from 0 to k-1 and we want to return k tuples with the indexes of the train and test folds the indexes are lists of integers of length respectively k-n_test and n_test
    test_folds = list(itertools.combinations(range(config.k),config.n_test_folds))
    print("num of test_folds combinations",len(test_folds))
    for fold_number, test_fold in enumerate(test_folds):
        # train_folds is a list list of indexes of the train folds and test is list of list of indexes of the test folds we have to flatten the lists and use those to vcat the series in folds to get the indexes of the train and test samples for each fold
        test_series = pl.concat([folds[i] for i in test_fold]).sort()
        train_series = pl.concat([folds[i] for i in range(config.k) if i not in test_fold]).sort()
        train_indexes.append(train_series.to_list())
        test_indexes.append(test_series.to_list())
        fold_numbers.append(fold_number)
    summary = CVSummary(
        cv_config=config,
        train_indexes=train_indexes,
        test_indexes=test_indexes,
        fold_numbers=fold_numbers,
    )
    return summary

def kfold_montecarlo(df: pl.DataFrame, config: KFoldConfig) -> CVSummary:
    df = check_add_cv_index(df,strict=True)
    cv_index = df["cv_index"].shuffle(seed=config.random_state)
    num_samples = cv_index.shape[0]
    fold_size = num_samples // config.k
    train_indexes = []
    test_indexes = []
    montecarlo_replicas = []
    for i in range(config.montecarlo_replicas):
        train_series = cv_index.sample(frac=(config.k-config.n_test_folds)/config.k,replace=False,seed=config.random_state+i)
        test_series = cv_index.filter(train_series,keep=False)
        train_indexes.append(train_series.to_list())
        test_indexes.append(test_series.to_list())
        montecarlo_replicas.append(i)
    summary = CVSummary(
        cv_config=config,
        train_indexes=train_indexes,
        test_indexes=test_indexes,
        replica_numbers =montecarlo_replicas,
    )
    return summary

def purged_combinatorial(df:pl.DataFrame, config: PurgedConfig) -> CVSummary:
    df = check_add_cv_index(df,strict=True)
    gdf = df.group_by(config.groups).agg(pl.col("cv_index")).select(pl.col([config.groups]+["cv_index"]))
    gdf_slices = slice_frame(gdf,config.k,shuffle=config.shuffle,explode=True)
    train_indexes = []
    test_indexes = []
    fold_numbers = []
    test_folds = list(itertools.combinations(range(config.k),config.n_test_folds))
    for fold_number, test_fold in enumerate(test_folds):
        test_series = pl.concat([gdf_slices[i] for i in test_fold]).sort()
        train_series = pl.concat([gdf_slices[i] for i in range(config.k) if i not in test_fold]).sort()
        train_indexes.append(train_series.to_list())
        test_indexes.append(test_series.to_list())
        fold_numbers.append(fold_number)
    summary = CVSummary(
        cv_config=config,
        train_indexes=train_indexes,
        test_indexes=test_indexes,
        fold_numbers=fold_numbers,
    )
    return summary

def purged_montecarlo(df:pl.DataFrame, config: PurgedConfig) -> CVSummary:
    df = check_add_cv_index(df,strict=True)
    gdf = df.group_by(config.groups).agg(pl.col("cv_index")).select(pl.col([config.groups]+["cv_index"]))
    train_indexes = []
    test_indexes = []
    montecarlo_replicas = []
    for i in range(config.montecarlo_replicas):
        gdf_slices = slice_frame(gdf,config.k,shuffle=True,explode=True)
        train_series = pl.concat(gdf_slices[:config.k-config.n_test_folds]).sort()
        test_series = pl.concat(gdf_slices[config.k-config.n_test_folds:]).sort()
        train_indexes.append(train_series.to_list())
        test_indexes.append(test_series.to_list())
        montecarlo_replicas.append(i)
    summary = CVSummary(
        cv_config=config,
        train_indexes=train_indexes,
        test_indexes=test_indexes,
        replica_numbers =montecarlo_replicas,
    )
    return summary

def stratified_combinatorial(df:pl.DataFrame, config: StratifiedConfig) -> CVSummary:
    k = config.k
    df = check_add_cv_index(df,strict=True)
    sdf = df.group_by(config.groups).agg(pl.col("cv_index"))
    if config.shuffle:
        sdf = sdf.with_columns(pl.col("cv_index").list.sample(fraction=1,shuffle=True))
    sliced = sdf.select(pl.col("cv_index").map_elements(lambda s: hacky_list_relative_slice(s,k)).alias("hacky_cv_index")).unnest("hacky_cv_index")
    train_indexes = []
    test_indexes = []
    fold_numbers = []
    test_folds = list(itertools.combinations(range(config.k),config.n_test_folds))
    for fold_number, test_fold in enumerate(test_folds):
        test_series=sliced.select(pl.concat_list([sliced["fold_{}".format(j)] for j in range(config.k) if j in test_fold]).alias("cv_index")).explode("cv_index")["cv_index"]
        train_series = sliced.select(pl.concat_list([sliced["fold_{}".format(j)] for j in range(config.k) if j not in test_fold]).alias("cv_index")).explode("cv_index")["cv_index"]
        train_indexes.append(train_series.to_list())
        test_indexes.append(test_series.to_list())
        fold_numbers.append(fold_number)
    summary = CVSummary(
        cv_config=config,
        train_indexes=train_indexes,
        test_indexes=test_indexes,
        fold_numbers=fold_numbers,
    )
    return summary

def stratified_montecarlo(df:pl.DataFrame, config: StratifiedConfig) -> CVSummary:
    k = config.k
    df = check_add_cv_index(df,strict=True)
    sdf = df.group_by(config.groups).agg(pl.col("cv_index"))
    if config.shuffle:
        sdf = sdf.with_columns(pl.col("cv_index").list.sample(fraction=1,shuffle=True))
    #instead of hackyrelative slice we can sampple the t
    train_indexes = []
    test_indexes = []
    montecarlo_replicas = []
    for i in range(config.montecarlo_replicas):
        traintest = sdf.select(pl.col("cv_index"),pl.col("cv_index").list.sample(fraction=config.n_test_folds/k).alias("test_index")).with_columns(pl.col("cv_index").list.set_difference(pl.col("test_index")))
        train_series = traintest.select("train_index").explode("train_index")["train_index"]
        test_series = traintest.select("test_index").explode("test_index")["test_index"]
        train_indexes.append(train_series.to_list())
        test_indexes.append(test_series.to_list())
        montecarlo_replicas.append(i)
    summary = CVSummary(
        cv_config=config,
        train_indexes=train_indexes,
        test_indexes=test_indexes,
        replica_numbers =montecarlo_replicas,
    )
    return summary

def cv_from_config(df:pl.DataFrame,config:BaseFoldConfig) -> CVSummary:
    if isinstance(config,KFoldConfig) and config.fold_mode.COMBINATORIAL:
        return kfold_combinatorial(df,config)
    elif isinstance(config,KFoldConfig) and config.fold_mode.MONTE_CARLO:
        return kfold_montecarlo(df,config)
    elif isinstance(config,PurgedConfig) and config.fold_mode.COMBINATORIAL:
        return purged_combinatorial(df,config)
    elif isinstance(config,PurgedConfig) and config.fold_mode.MONTE_CARLO:
        return purged_montecarlo(df,config)
    elif isinstance(config,StratifiedConfig) and config.fold_mode.COMBINATORIAL:
        return stratified_combinatorial(df,config)
    elif isinstance(config,StratifiedConfig) and config.fold_mode.MONTE_CARLO:
        return stratified_montecarlo(df,config)
    else:
        raise ValueError(f"Unsupported fold configuration: {config}")

def train_test_val(df:pl.DataFrame,train_idx:pl.DataFrame,val_idx:pl.DataFrame,test_idx:pl.DataFrame) -> Tuple[pl.DataFrame,pl.DataFrame,pl.DataFrame]:
    print("training idx",train_idx)
    df_train = df.filter(pl.col("cv_index").is_in(train_idx["cv_index"]))
    df_val = df.filter(pl.col("cv_index").is_in(val_idx["cv_index"]))
    df_test = df.filter(pl.col("cv_index").is_in(test_idx["cv_index"]))
    return df_train,df_val,df_test

def generate_nested_cv(df_idx:pl.DataFrame,task_config:PredictConfig) -> Generator[PipelineInput,None,None]:
    cv_config = task_config.cv_config
    input_config = task_config.input_config
    classifiers_config = task_config.classifiers_config
    for r_o in range(cv_config.outer_replicas):
        outer_cv = cv_from_config(df_idx, cv_config.outer)
        #Outer Folds -- this is an instance of an outer cross-validation fold
        for k_o, (dev_idx_o,test_idx_o) in enumerate(outer_cv.yield_splits()):
            df_test_idx_o = df_idx.filter(pl.col("cv_index").is_in(test_idx_o))
            df_dev_idx_o = df_idx.filter(pl.col("cv_index").is_in(dev_idx_o))
            #Inner Replicas
            for r_i in range(cv_config.inner_replicas):
                inner_cv = cv_from_config(df_dev_idx_o, cv_config.inner)
                #Inner Folds -- this is an instance of an inner cross-validation fold
                for k_i,(train_idx_i,val_idx_i) in enumerate(inner_cv.yield_splits()):
                    df_val_idx_o_i = df_idx.filter(pl.col("cv_index").is_in(val_idx_i))
                    df_train_idx_o_i = df_idx.filter(pl.col("cv_index").is_in(train_idx_i))
                    n_train = df_train_idx_o_i.shape[0]
                    n_val = df_val_idx_o_i.shape[0]
                    n_test = df_test_idx_o.shape[0]
                    print(f"For outer replica {r_o}, outer fold {k_o}, inner replica {r_i}, inner fold {k_i}: train {n_train}, val {n_val}, test {n_test} samples.")
                    #Feature types loop
                    for feature_index,feature_set in enumerate(input_config.feature_sets):
                        for classifier in classifiers_config.classifiers:
                            yield PipelineInput(train_idx=df_train_idx_o_i,val_idx= df_val_idx_o_i,test_idx= df_test_idx_o,feature_index= feature_index,cls_config= classifier,input_config=input_config)

---

## cv test

import logfire

logfire.install_auto_tracing(modules=['cynde'])
logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record='all'))



from cynde.functional.distributed_cv import train_nested_cv_from_np_modal, cv_stub, preprocess_np_modal
import cynde.functional as cf
import os
import polars as pl
from typing import List, Optional, Tuple, Generator
import time
from cynde.functional.predict.types import PredictConfig, BaseClassifierConfig,StratifiedConfig,Feature,FeatureSet,NumericalFeature, CategoricalFeature,EmbeddingFeature, InputConfig, ClassifierConfig, LogisticRegressionConfig, RandomForestClassifierConfig, HistGradientBoostingClassifierConfig, CVConfig
from cynde.functional.predict.preprocess import convert_utf8_to_enum, check_add_cv_index, preprocess_inputs, load_preprocessed_features
from cynde.functional.predict.cv import stratified_combinatorial
from cynde.functional.predict.classify import create_pipeline ,train_nested_cv

from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline






def load_minihermes_data(data_path: str = r"C:\Users\Tommaso\Documents\Dev\Cynde\cache\OpenHermes-2.5_embedded.parquet") -> pl.DataFrame:
    return pl.read_parquet(data_path)

df = load_minihermes_data()
df = convert_utf8_to_enum(df, threshold=0.2)
df = check_add_cv_index(df,strict=False)
print(df.columns)

feature_set_small_data = {"embeddings":[{"column_name":"conversations_text-embedding-3-small_embeddings",
                                         "name":"feature set for the smaller oai embeddings",
                                         "embedder": "text-embedding-3-small_embeddings",
                                         "embedding_size":1536}]}
feature_set_large_data = {"embeddings":[{"column_name":"conversations_text-embedding-3-large_embeddings",
                                         "name":"feature set for the larger oai embeddings",
                                        "embedder": "text-embedding-3-large_embeddings",
                                         "embedding_size":3072}]}

input_config_data = {"feature_sets":[feature_set_small_data,feature_set_large_data],
                        "target_column":"target",
                        "save_folder":"C:/Users/Tommaso/Documents/Dev/Cynde/cynde_mount/"}

input_config = InputConfig.model_validate(input_config_data,context={"df":df})
print("Input config:")
print(input_config)
preprocess_inputs(df, input_config)

classifiers_config = ClassifierConfig(classifiers=[RandomForestClassifierConfig(n_estimators=100),RandomForestClassifierConfig(n_estimators=500)])
print("Classifiers config:")
print(classifiers_config)
groups = ["target"]
cv_config = CVConfig(inner= StratifiedConfig(groups=groups,k=5),
                     inner_replicas=1,
                     outer = StratifiedConfig(groups=groups,k=5),
                        outer_replicas=1)
print("CV config:")
print(cv_config)

task = PredictConfig(input_config=input_config, cv_config=cv_config, classifiers_config=classifiers_config)

train_nested_cv(df,task)



---

## distributed

import modal
from modal import Image
from typing import Tuple
from cynde.functional.predict.classify import predict_pipeline
from cynde.functional.predict.types import PipelineInput,PipelineResults,PredictConfig
from cynde.functional.predict.preprocess import load_preprocessed_features,check_add_cv_index
from cynde.functional.predict.cv import train_test_val,generate_nested_cv
from cynde.functional.predict.classify import create_pipeline ,evaluate_model



app = modal.App("distributed_cv")
    
datascience_image = (
    Image.debian_slim(python_version="3.12.1")
    .apt_install("git")
    .pip_install("polars","scikit-learn","openai","tiktoken")#, force_build=True)
    
    .run_commands("git clone https://github.com/Neural-Dragon-AI/Cynde/")
    .env({"CYNDE_DIR": "/opt/cynde"})
    .run_commands("cd Cynde && pip install -r requirements.txt && pip install .")
)
with datascience_image.imports():
    import polars as pl
    import sklearn as sk
    import cynde as cy


#define the distributed classification method
@app.function(image=datascience_image, mounts=[modal.Mount.from_local_dir(r"C:\Users\Tommaso\Documents\Dev\Cynde\cynde_mount", remote_path="/root/cynde_mount")])
def predict_pipeline_distributed(pipeline_input:PipelineInput) -> Tuple[pl.DataFrame,pl.DataFrame,float,float]:
    input_config = pipeline_input.input_config
    feature_set = input_config.feature_sets[pipeline_input.feature_index]
    df_fold = load_preprocessed_features(input_config,pipeline_input.feature_index,remote=True)
    print(df_fold)
    df_train,df_val,df_test = train_test_val(df_fold,pipeline_input.train_idx,pipeline_input.val_idx,pipeline_input.test_idx)
    print(df_train)
    pipeline = create_pipeline(df_train, feature_set, pipeline_input.cls_config)
    print(pipeline)
    pipeline.fit(df_train,df_train["target"])
    train_predictions, train_accuracy, train_mcc = evaluate_model(pipeline, df_train, df_train["target"])
    val_predictions,val_accuracy, val_mcc = evaluate_model(pipeline, df_val, df_val["target"])
    test_predictions,test_accuracy,test_mcc = evaluate_model(pipeline, df_test, df_test["target"])
    return PipelineResults(train_predictions = train_predictions,
                           val_predictions=val_predictions,
                           test_predictions=test_predictions,
                           train_accuracy=train_accuracy,
                           train_mcc=train_mcc,
                           val_accuracy=val_accuracy,
                           val_mcc=val_mcc,
                           test_accuracy=test_accuracy,
                           test_mcc=test_mcc)

def train_nested_cv_distributed(df:pl.DataFrame,task_config:PredictConfig) -> pl.DataFrame:
    """ Deploy a CV training pipeline to Modal, it requires a df with cv_index column and the features set to have already pre-processed and cached 
    1) Validate the input_config and check if the preprocessed features are present locally 
    2) create a generator that yields the modal path to the features and targets frames as well as the scikit pipeline object 
    3) execute through a modal starmap a script that fit end eval each pipeline on each feature set and return the results
    4) collect and aggregate the results locally and save and return the results
    """
    #validate the inputs and check if the preprocessed features are present locally
    df = check_add_cv_index(df,strict=True)
    
    
    #extract the subset of columns necessary for constructing the cross validation folds 
    unique_groups = list(set(task_config.cv_config.inner.groups + task_config.cv_config.outer.groups))
    df_idx = df.select(pl.col("cv_index"),pl.col(unique_groups))

    nested_cv = generate_nested_cv(df_idx,task_config)
    all_results = []
    for result in predict_pipeline_distributed.map(list(nested_cv)):
        all_results.append(result)
    re_validated_results = []
    for result in all_results:
        re_validated_results.append(PipelineResults.model_validate(result))
    print("Finished!! " ,len(all_results))

---

## preprocess

import polars as pl
import numpy as np
from typing import Optional, Tuple
from cynde.functional.predict.types import InputConfig,FeatureSet
import os

def convert_utf8_to_enum(df: pl.DataFrame, threshold: float = 0.2) -> pl.DataFrame:
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1 (exclusive).")

    for column in df.columns:
        if df[column].dtype == pl.Utf8 and len(df[column]) > 0:
            unique_values = df[column].unique()
            unique_ratio = len(unique_values) / len(df[column])

            if unique_ratio <= threshold:
                enum_dtype = pl.Enum(unique_values.to_list())
                df = df.with_columns(df[column].cast(enum_dtype))
            else:
                print(f"Column '{column}' has a high ratio of unique values ({unique_ratio:.2f}). Skipping conversion to Enum.")
        elif df[column].dtype == pl.Utf8 and len(df[column]) == 0:
            print(f"Column '{column}' is empty. Skipping conversion to Enum.")

    return df

def convert_enum_to_physical(df: pl.DataFrame) -> pl.DataFrame:
    df_physical = df.with_columns(
        [pl.col(col).to_physical() for col in df.columns if df[col].dtype == pl.Enum]
    )
    return df_physical

def check_add_cv_index(df:pl.DataFrame,strict:bool=False) -> Optional[pl.DataFrame]:
    if "cv_index" not in df.columns and not strict:
        df = df.with_row_index(name="cv_index")
    elif "cv_index" not in df.columns and strict:
        raise ValueError("cv_index column not found in the DataFrame.")
    return df

def preprocess_inputs(df: pl.DataFrame, input_config: InputConfig):
    """ Saves .parquet for each feature set in input_config """
    save_folder = input_config.save_folder
    for feature_set in input_config.feature_sets:
        column_names = feature_set.column_names()
        feature_set_df = df.select(pl.col("cv_index"),pl.col("target"),pl.col(column_names))
        print(f"selected columns: {feature_set_df.columns}")
        #explodes all the embedding columns into list of columns
        # for feature in feature_set.embeddings:
        #     print(f"Converting {feature.column_name} of type {df[feature.column_name].dtype} to a list of columns.")
        #     feature_set_df = map_list_to_cols(feature_set_df,feature.column_name)

        save_name = feature_set.joined_names()
        save_path = os.path.join(save_folder, f"{save_name}.parquet")
        feature_set_df.write_parquet(save_path)

def load_preprocessed_features(input_config:InputConfig,feature_set_id:int,convert_embeddings:bool=True, remote:bool = False) -> pl.DataFrame:
    """ Loads the the train,val and test df for a specific feature set fold """
    folder = input_config.save_folder if not remote else input_config.remote_folder
    print("""Loading from folder: {}""".format(folder))

    feature_set = input_config.feature_sets[feature_set_id]
    file_name = feature_set.joined_names()
    if file_name is None:
        raise ValueError(f"Feature set {feature_set_id} not found in input config.")
    file_path = os.path.join(folder, f"{file_name}.parquet")
    df = pl.read_parquet(file_path)
    if convert_embeddings:
        for feature in feature_set.embeddings:
            print(f"Converting {feature.column_name} of type {df[feature.column_name].dtype} to a list of columns.")
            df = map_list_to_cols(df,feature.column_name)
    return df

def validate_preprocessed_inputs(input_config:InputConfig) -> None:
    """ Validates the preprocessed input config checking if the .parquet files are present """
    path = input_config.save_folder
    for feature_set in input_config.feature_sets:
        save_name = feature_set.joined_names()
        save_path = os.path.join(path, f"{save_name}.parquet")
        if not os.path.exists(save_path):
            raise ValueError(f"Preprocessed feature set '{save_name}' not found at '{save_path}'.")


def map_list_to_cols(df:pl.DataFrame, list_column:str) -> pl.DataFrame:
    """ Maps a list column to a DataFrame """
    width = len(df[list_column][0])
    return df.with_columns(pl.col(list_column).list.get(i).alias(f"{list_column}_{i}") for i in range(width)).select(pl.all().exclude(list_column))


---

## types

from enum import Enum
import polars as pl
from pydantic import BaseModel, ValidationInfo, model_validator,Field,ValidationInfo, field_validator

from enum import Enum
from typing import Optional, Union, Dict, Literal, Any, List, Tuple, Type, TypeVar, Generator


from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer, OneHotEncoder



class ScalerType(str, Enum):
    STANDARD_SCALER = "StandardScaler"
    MIN_MAX_SCALER = "MinMaxScaler"
    MAX_ABS_SCALER = "MaxAbsScaler"
    ROBUST_SCALER = "RobustScaler"
    POWER_TRANSFORMER = "PowerTransformer"
    QUANTILE_TRANSFORMER = "QuantileTransformer"
    NORMALIZER = "Normalizer"

class Feature(BaseModel):
    column_name: str
    name: str
    description: Optional[str] = None

    @field_validator("column_name")
    @classmethod
    def column_in_df(cls, v: str, info: ValidationInfo):
        column_name = v
        context = info.context
        if context:
            df = context.get("df",pl.DataFrame())
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        return v


class NumericalFeature(Feature):
    scaler_type: ScalerType = Field(ScalerType.STANDARD_SCALER, description="The type of scaler to apply to the numerical feature.")

    def get_scaler(self):
        scaler_map = {
            ScalerType.STANDARD_SCALER: StandardScaler(),
            ScalerType.MIN_MAX_SCALER: MinMaxScaler(),
            ScalerType.MAX_ABS_SCALER: MaxAbsScaler(),
            ScalerType.ROBUST_SCALER: RobustScaler(),
            ScalerType.POWER_TRANSFORMER: PowerTransformer(),
            ScalerType.QUANTILE_TRANSFORMER: QuantileTransformer(),
            ScalerType.NORMALIZER: Normalizer(),
        }
        return scaler_map[self.scaler_type]
    
    @field_validator("column_name")
    @classmethod
    def column_correct_type(cls, v: str, info: ValidationInfo):
        column_name = v
        context = info.context
        if context:
            df = context.get("df",pl.DataFrame())
            if df[column_name].dtype not in [pl.Boolean, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64, pl.Decimal]:
                current_dtype = df[column_name].dtype
                raise ValueError(f"Column '{column_name}'  with dtype {current_dtype} must be of a numeric type (Boolean, Integer, Unsigned Integer, Float, or Decimal) .")

        return v


class EmbeddingFeature(NumericalFeature):
    embedder: str = Field("text-embedding-3-small", description="The embedder model that generated the vector.")
    embedding_size:int = Field(1536, description="The size of the embedding vector.")

    @field_validator("column_name")
    @classmethod
    def column_correct_type(cls, v: str, info: ValidationInfo):
        column_name = v
        context = info.context
        if context:
            df = context.get("df",pl.DataFrame())
            if df[column_name].dtype not in [pl.List(pl.Float32), pl.List(pl.Float64)]:
                current_dtype = df[column_name].dtype
                raise ValueError(f"Column '{column_name}'  with dtype {current_dtype} must be of type pl.List(pl.Float32) or pl.List(pl.Float64).")
        return v

class CategoricalFeature(Feature):
    one_hot_encoding: bool = Field(True, description="Whether to apply one-hot encoding to the categorical feature.")

    @field_validator("column_name")
    @classmethod
    def column_correct_type(cls, v: str, info: ValidationInfo):
        column_name = v
        context = info.context
        if context:
            df = context.get("df",pl.DataFrame())
            if df[column_name].dtype not in [
                pl.Utf8,
                pl.Categorical,
                pl.Enum,
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            ]:
                current_dtype = df[column_name].dtype
                raise ValueError(
                    f"Column '{column_name}' with dtype {current_dtype}  must be of type pl.Utf8, pl.Categorical, pl.Enum, or an integer type."
                )
        return v

class FeatureSet(BaseModel):
    numerical: List[NumericalFeature] = []
    embeddings: List[EmbeddingFeature] = []
    categorical: List[CategoricalFeature] = []


    def all_features(self):
        return self.numerical + self.embeddings + self.categorical
    def column_names(self):
        return [feature.column_name for feature in self.all_features()]
    def joined_names(self):
        return "_".join(sorted(self.column_names()))

class InputConfig(BaseModel):
    feature_sets: List[FeatureSet]
    target_column: str = Field("target", description="The target column to predict.")
    remote_folder: str = Field("/root/cynde_mount", description="The remote folder to save the preprocessed features.")

    save_folder: Optional[str] = None



class ClassifierName(str, Enum):
    LOGISTIC_REGRESSION = "LogisticRegression"
    RANDOM_FOREST = "RandomForestClassifier"
    HIST_GRADIENT_BOOSTING = "HistGradientBoostingClassifier"

class BaseClassifierConfig(BaseModel):
    classifier_name: ClassifierName
    

class LogisticRegressionConfig(BaseClassifierConfig):
    classifier_name: Literal[ClassifierName.LOGISTIC_REGRESSION] = Field(ClassifierName.LOGISTIC_REGRESSION)
    n_jobs: int = Field(-1, description="Number of CPU cores to use.")
    penalty: str = Field("l2", description="Specify the norm of the penalty.")
    dual: bool = Field(False, description="Dual or primal formulation.")
    tol: float = Field(1e-4, description="Tolerance for stopping criteria.")
    C: float = Field(1.0, description="Inverse of regularization strength.")
    fit_intercept: bool = Field(True, description="Specifies if a constant should be added to the decision function.")
    intercept_scaling: float = Field(1, description="Scaling factor for the constant.")
    class_weight: Optional[Union[str, Dict[Any, float]]] = Field(None, description="Weights associated with classes.")
    random_state: Optional[int] = Field(None, description="Seed for random number generation.")
    solver: str = Field("lbfgs", description="Algorithm to use in the optimization problem.")
    max_iter: int = Field(100, description="Maximum number of iterations.")
    multi_class: str = Field("auto", description="Approach for handling multi-class targets.")
    verbose: int = Field(0, description="Verbosity level.")
    warm_start: bool = Field(False, description="Reuse the solution of the previous call to fit.")
    l1_ratio: Optional[float] = Field(None, description="Elastic-Net mixing parameter.")

class RandomForestClassifierConfig(BaseClassifierConfig):
    classifier_name: Literal[ClassifierName.RANDOM_FOREST] = Field(ClassifierName.RANDOM_FOREST)
    n_jobs: int = Field(-1, description="Number of CPU cores to use.")
    n_estimators: int = Field(100, description="The number of trees in the forest.")
    criterion: str = Field("gini", description="The function to measure the quality of a split.")
    max_depth: Optional[int] = Field(None, description="The maximum depth of the tree.")
    min_samples_split: Union[int, float] = Field(2, description="The minimum number of samples required to split an internal node.")
    min_samples_leaf: Union[int, float] = Field(1, description="The minimum number of samples required to be at a leaf node.")
    min_weight_fraction_leaf: float = Field(0.0, description="The minimum weighted fraction of the sum total of weights required to be at a leaf node.")
    max_features: Union[str, int, float] = Field("sqrt", description="The number of features to consider when looking for the best split.")
    max_leaf_nodes: Optional[int] = Field(None, description="Grow trees with max_leaf_nodes in best-first fashion.")
    min_impurity_decrease: float = Field(0.0, description="A node will be split if this split induces a decrease of the impurity greater than or equal to this value.")
    bootstrap: bool = Field(True, description="Whether bootstrap samples are used when building trees.")
    oob_score: bool = Field(False, description="Whether to use out-of-bag samples to estimate the generalization score.")
    
    random_state: Optional[int] = Field(None, description="Seed for random number generation.")
    verbose: int = Field(0, description="Verbosity level.")
    warm_start: bool = Field(False, description="Reuse the solution of the previous call to fit and add more estimators to the ensemble.")
    class_weight: Optional[Union[str, Dict[Any, float]]] = Field(None, description="Weights associated with classes.")
    ccp_alpha: float = Field(0.0, description="Complexity parameter used for Minimal Cost-Complexity Pruning.")
    max_samples: Optional[Union[int, float]] = Field(None, description="If bootstrap is True, the number of samples to draw from X to train each base estimator.")
    monotonic_cst: Optional[Dict[str, int]] = Field(None, description="Monotonic constraint to enforce on each feature.")

class HistGradientBoostingClassifierConfig(BaseClassifierConfig):
    classifier_name: Literal[ClassifierName.HIST_GRADIENT_BOOSTING] = Field(ClassifierName.HIST_GRADIENT_BOOSTING)
    loss: str = Field("log_loss", description="The loss function to use in the boosting process.")
    learning_rate: float = Field(0.1, description="The learning rate, also known as shrinkage.")
    max_iter: int = Field(100, description="The maximum number of iterations of the boosting process.")
    max_leaf_nodes: int = Field(31, description="The maximum number of leaves for each tree.")
    max_depth: Optional[int] = Field(None, description="The maximum depth of each tree.")
    min_samples_leaf: int = Field(20, description="The minimum number of samples per leaf.")
    l2_regularization: float = Field(0.0, description="The L2 regularization parameter.")
    max_features: Union[str, int, float] = Field(1.0, description="Proportion of randomly chosen features in each and every node split.")
    max_bins: int = Field(255, description="The maximum number of bins to use for non-missing values.")
    categorical_features: Optional[Union[str, List[int], List[bool]]] = Field("warn", description="Indicates the categorical features.")
    monotonic_cst: Optional[Dict[str, int]] = Field(None, description="Monotonic constraint to enforce on each feature.")
    interaction_cst: Optional[Union[str, List[Tuple[int, ...]]]] = Field(None, description="Specify interaction constraints, the sets of features which can interact with each other in child node splits.")
    warm_start: bool = Field(False, description="Reuse the solution of the previous call to fit and add more estimators to the ensemble.")
    early_stopping: Union[str, bool] = Field("auto", description="Whether to use early stopping to terminate training when validation score is not improving.")
    scoring: Optional[str] = Field("loss", description="Scoring parameter to use for early stopping.")
    validation_fraction: float = Field(0.1, description="Proportion of training data to set aside as validation data for early stopping.")
    n_iter_no_change: int = Field(10, description="Used to determine when to stop if validation score is not improving.")
    tol: float = Field(1e-7, description="The absolute tolerance to use when comparing scores.")
    verbose: int = Field(0, description="Verbosity level.")
    random_state: Optional[int] = Field(None, description="Seed for random number generation.")
    class_weight: Optional[Union[str, Dict[Any, float]]] = Field(None, description="Weights associated with classes.")

class ClassifierConfig(BaseModel):
    classifiers: List[Union[LogisticRegressionConfig, RandomForestClassifierConfig, HistGradientBoostingClassifierConfig]]


class FoldMode(str, Enum):
    COMBINATORIAL = "Combinatorial"
    MONTE_CARLO = "MonteCarlo"

class BaseFoldConfig(BaseModel):
    k: int = Field(5, description="Number of folds. Divides the data into k equal parts. last k can be smaller or larger than the rest depending on the // of the data by k." )
    n_test_folds: int = Field(1, description="Number of test folds to use for cross-validation. Must be strictly less than k. if the fold mode is montecarlo they are sampled first and then the rest are used for training. If the fold mode is combinatorial the all symmetric combinations n_test out of k are sampled.")
    fold_mode: FoldMode = Field(FoldMode.COMBINATORIAL, description="The mode to use for splitting the data into folds. Combinatorial splits the data into k equal parts, while Monte Carlo randomly samples the k equal parts without replacement.")
    shuffle: bool = Field(True, description="Whether to shuffle the data before splitting.")
    random_state: Optional[int] = Field(None, description="Seed for random number generation. In the case of montecarlo cross-validation at each replica the seed is increased by 1 mantaining replicability while ensuring that the samples are different.")
    montecarlo_replicas: int = Field(5, description="Number of random replicas to use for montecarlo cross-validation.")

class KFoldConfig(BaseFoldConfig):
    pass

class StratificationMode(str, Enum):
    PROPORTIONAL = "Proportional"
    UNIFORM_STRICT = "UniformStrict"
    UNIFORM_RELAXED = "UniformRelaxed"

class StratifiedConfig(BaseFoldConfig):
    groups: List[str] = Field([], description="The df column(s) to use for stratification. They will be used for a group-by operation to ensure that the stratification is done within each group.")
    strat_mode : StratificationMode = Field(StratificationMode.PROPORTIONAL, description="The mode to use for stratification. Proportional ensures that the stratification is done within each group mantaining the original proportion of each group in the splits, this is done by first grouping and then breaking each group inot k equal parts, this ensure all the samples in each group are in train and test with the same proprtion. Uniform instead ensures that each group has the same number of samples in each train and test fold, this is not compatible with the proportional mode.")
    group_size : Optional[int] = Field(None, description="The number of samples to use for each group in the stratificaiton it will only be used if the strat_mode is uniform or uniform relaxed. If uniform relaxed is used the group size will be used as a target size for each group but if a group has less samples than the target size it will be used as is. If uniform strict is used the group_size for all groups will be forced to the min(group_size, min_samples_in_group).")

class PurgedConfig(BaseFoldConfig):
    groups: List[str] = Field([], description="The df column(s) to use for purging. They will be used for a group-by operation to ensure that the purging at the whole group level. K is going to used to determine the fraction of groups to purge from train and restrict to test. When the mode is montecarlo the groups are sampled first and then the rest are used for training. If the fold mode is combinatorial the all symmetric combinations n_test out of k groups partitions are sampled")

class CVConfig(BaseModel):
    inner: Union[KFoldConfig,StratifiedConfig,PurgedConfig]
    inner_replicas: int = Field(1, description="Number of random replicas to use for inner cross-validation.")
    outer: Optional[Union[KFoldConfig,StratifiedConfig,PurgedConfig]] = None
    outer_replicas: int = Field(1, description="Number of random replicas to use for outer cross-validation.")

class PredictConfig(BaseModel):
    cv_config: CVConfig
    input_config: InputConfig
    classifiers_config: ClassifierConfig

class CVSummary(BaseModel):
    cv_config: Union[KFoldConfig,StratifiedConfig,PurgedConfig] = Field(description="The cross-validation configuration. Required for the summary.")
    train_indexes: List[List[int]] = Field(description="The indexes of the training samples for each fold or replica.")
    test_indexes: List[List[int]] = Field(description="The indexes of the testing samples for each fold or replica.")
    fold_numbers: Optional[List[int]] = Field(None, description="The fold number for each sample. Used when the fold mode is combinatorial.")
    replica_numbers : Optional[List[int]] = Field(None, description="The replica number for each sample. Used when the fold mode is montecarlo.")
    
    def yield_splits(self) -> Generator[Tuple[pl.Series, pl.Series], None, None]:
        for train_idx, test_idx in zip(self.train_indexes, self.test_indexes):
            train_series = pl.Series(train_idx)
            test_series = pl.Series(test_idx)
            yield train_series, test_series

class PipelineInput(BaseModel):
    train_idx:pl.DataFrame
    val_idx:pl.DataFrame
    test_idx:pl.DataFrame
    feature_index:int
    cls_config:BaseClassifierConfig
    input_config : InputConfig

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

class PipelineResults(BaseModel):
    train_predictions:pl.DataFrame
    val_predictions:pl.DataFrame
    test_predictions:pl.DataFrame
    train_accuracy:float
    train_mcc:float
    val_accuracy:float
    val_mcc:float
    test_accuracy:float
    test_mcc:float

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"





---

## prompt

from typing import List, Union
import polars as pl

def prompt(df: pl.DataFrame, fstring: str, expressions: List[Union[pl.Expr, str]], prompt_name: str, context: str = 'with_columns') -> pl.DataFrame:
    """
    Dynamically generates prompts based on the given format string and expressions, 
    and either adds them as a new column to the DataFrame or selects them based on the specified context.

    Parameters:
    - df: The Polars DataFrame to which the prompts will be added or from which data will be selected.
    - fstring: A format string with placeholders for the expressions. If a plain string value is to be included, 
               it will be converted to a Polars expression.
    - expressions: A list of Polars expressions or string literals. Each expression must result in either a scalar value 
                   or a list of values all having the same length. When using 'with_columns' context, the expressions 
                   must return lists of the same length as the full DataFrame.
    - prompt_name: The name of the new column that will contain the generated prompts.
    - context: A string indicating the operation context. Valid values are 'with_columns' and 'select'.
               'with_columns' appends the generated prompts as a new column, requiring list results to match
               the DataFrame length. 'select' creates a new DataFrame from the generated prompts, potentially
               alongside other specified columns.

    Returns:
    - A DataFrame with the added prompts column if 'with_columns' is used, or a new DataFrame with selected columns
      if 'select' is used. The result of each expression used in the formatting must result in either a scalar 
      value or a list of values all having the same length, especially for 'with_columns' context.
    """
    # Convert string values in expressions to Polars expressions
    expressions = [pl.lit(expr) if isinstance(expr, str) else expr for expr in expressions]

    # Validate inputs
    if not isinstance(df, pl.DataFrame):
        raise ValueError("df must be a Polars DataFrame.")
    if not isinstance(fstring, str):
        raise ValueError("fstring must be a string.")
    if not all(isinstance(expr, pl.Expr) for expr in expressions):
        raise ValueError("All items in expressions must be Polars Expr or string literals converted to Polars expressions.")
    if not isinstance(prompt_name, str):
        raise ValueError("prompt_name must be a string.")
    if context not in ['with_columns', 'select']:
        raise ValueError("context must be either 'with_columns' or 'select'.")

    # Validate the number of placeholders matches the number of expressions
    placeholders_count = fstring.count("{}")
    if placeholders_count != len(expressions):
        raise ValueError(f"The number of placeholders in fstring ({placeholders_count}) does not match the number of expressions ({len(expressions)}).")

    # Use pl.format to generate the formatted expressions
    formatted_expr = pl.format(fstring, *expressions).alias(prompt_name)
    
    # Apply the context-specific operation
    if context == 'with_columns':
        # Append the generated prompt as a new column
        return df.with_columns(formatted_expr)
    else:  # context == 'select'
        # Create a new DataFrame with only the generated prompts or alongside other specified columns
        return df.select(formatted_expr)


---

## results

import polars as pl
from cynde.functional.cv import get_fold_name_cv
from cynde.functional.classify import get_hp_classifier_name, get_pred_column_name, get_input_name
from typing import List, Dict, Union, Tuple, Any

def results_summary(results:pl.DataFrame,by_test_fold:bool=False) -> pl.DataFrame:
    groups = [ "classifier","classifier_hp","input_features_name"]
    if by_test_fold:
        groups += ["r_outer","r_inner"]
       
    summary = results.group_by(
   groups).agg(
    pl.col(["mcc_train","mcc_val","mcc_test"]).mean(),
     pl.col(["accuracy_train","accuracy_val","accuracy_test"]).mean(),
    pl.len().alias("n")).sort("mcc_val",descending=True)
    return summary




def get_predictions(joined_df:pl.DataFrame,
                    cv_type: Tuple[str, str],
                    inputs: List[Dict[str, Union[List[str], List[List[str]]]]],
                    models: Dict[str, List[Dict[str, Any]]],
                    group_outer: List[str],
                    k_outer: int,
                    group_inner: List[str],
                    k_inner: int,
                    r_outer: int = 1,
                    r_inner: int = 1,) -> pl.DataFrame:
    outs = []
    
    for r_o in range(r_outer):
        for k_o in range(k_outer):
            for r_i in range(r_inner):
                for k_i in range(k_inner):
                    fold_name = get_fold_name_cv(group_outer, cv_type, r_o,k_o,group_inner,r_i,k_i)
                    for input_feature in inputs:
                        input_name = get_input_name(input_feature)
                        for model, hp_list in models.items():
                            for hp in hp_list:
                                hp_name = get_hp_classifier_name(hp)
                                pred_col_name = get_pred_column_name(fold_name, input_name, model, hp_name)
                                if pred_col_name not in joined_df.columns:
                                    raise ValueError(f"Column {pred_col_name} not found in the joined_df")
                                outs.append((fold_name,pred_col_name))
    return outs

def get_all_predictions_by_inputs_model(joined_df:pl.DataFrame,
                    cv_type: Tuple[str, str],
                    inputs: List[Dict[str, Union[List[str], List[List[str]]]]],
                    models: Dict[str, List[Dict[str, Any]]],
                    group_outer: List[str],
                    k_outer: int,
                    group_inner: List[str],
                    k_inner: int,
                    r_outer: int = 1,
                    r_inner: int = 1,)  :
    
    for input_feature in inputs:
                        input_name = get_input_name(input_feature)
                        for model, hp_list in models.items():
                            for hp in hp_list:
                                hp_name = get_hp_classifier_name(hp)
                                pred_cols_by_model =[]
                                for r_o in range(r_outer):
                                    for k_o in range(k_outer):
                                        for r_i in range(r_inner):
                                            for k_i in range(k_inner):
                                                fold_name = get_fold_name_cv(group_outer, cv_type, r_o,k_o,group_inner,r_i,k_i)
                                                pred_col_name = get_pred_column_name(fold_name, input_name, model, hp_name)
                                                if pred_col_name not in joined_df.columns:
                                                    raise ValueError(f"Column {pred_col_name} not found in the joined_df")
                                                pred_cols_by_model.append((fold_name,pred_col_name))
                                yield input_name,model,hp_name,pred_cols_by_model

---

## init

# This is the __init__.py file for the package.


---

## init

# This is the __init__.py file for the package.


---

## embedders

from openai import Client
import time
from typing import List,Optional


def get_embedding_single(text:str, model:str="text-embedding-ada-002", client: Optional[Client]=None):
    if client is None:
        client = Client()
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_embedding_list(text_list:List[str], model:str="text-embedding-ada-002", batch_size=100,client: Optional[Client]=None):
    #count time for processing
    if client is None:
        client = Client()
    start = time.time()
    embeddings = []
    if len(text_list) < batch_size:
        print(f"Processing {len(text_list)} chunks of text in a single batch")
        # If the list is smaller than the batch size, process it in one go
        batch_embeddings = client.embeddings.create(input=text_list, model=model).data
        embeddings.extend([item.embedding for item in batch_embeddings])
    else:
        print(f"Processing {len(text_list)} chunks of text in batches")
        # Process the list in batches
        #get number of batches
        for i in range(0, len(text_list), batch_size):
            
            batch = text_list[i:i + batch_size]
            batch_embeddings = client.embeddings.create(input=batch, model=model).data
            embeddings.extend([item.embedding for item in batch_embeddings])
            print(f"Processed  {i} cunks of text out of {len(text_list)}")
    print(f"Embedding Processing took {time.time() - start} seconds")
    return embeddings



---

## init

# This is the __init__.py file for the package.


---

## expressions

import polars as pl

def list_struct_to_string(col_name: str, separator: str = " ") -> pl.Expr:
    return pl.col(col_name).list.eval(pl.element().struct.json_encode()).list.join(separator=separator).alias(f"str_{col_name}")


---

## regex

import re

def format_string_to_regex(fstring, var_names):
    # Escape regex special characters in the static parts of the format string
    escaped_string = re.escape(fstring)
    
    # Prepare a regex pattern for named capturing groups
    for var_name in var_names:
        # Replace the first occurrence of escaped "{}" with a named capturing group
        escaped_string = re.sub(r'\\\{\\\}', f'(?P<{var_name}>[^_]+)', escaped_string, count=1)
    
    return escaped_string

def parse_format_string(input_string, fstring, var_names):
    # Generate the regex pattern from the format string and variable names
    regex_pattern = format_string_to_regex(fstring, var_names)
    
    # Attempt to match the regex pattern against the input string
    match = re.search(regex_pattern, input_string)
    
    if match:
        # Extract matched values into a dictionary using the variable names
        result = {var_name: match.group(var_name) for var_name in var_names}
        return result
    else:
        # Return None or an empty dictionary if no match is found
        return None

---

