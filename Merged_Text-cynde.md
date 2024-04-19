# Combined Text Dir from cynde

- Full filepath to the merged directory: `C:\Users\Tommaso\Documents\Dev\Cynde\cynde`

- Created: `2024-04-11T13:09:18.085443`

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
set_directories(root_dir)

---

## classify

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,  matthews_corrcoef
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import polars as pl
import time
from typing import Tuple, Optional, List, Dict, Union, Any
import os

def get_hp_classifier_name(classifier_hp:dict) -> str:
    classifier_hp_name = "_".join([f"{key}_{value}" for key,value in classifier_hp.items()])
    return classifier_hp_name

def get_pred_column_name(fold_name:str,input_features_name:str,classifier:str,classifier_hp_name:str) -> str:
    return  "{}_{}_{}_{}_y_pred".format(fold_name,input_features_name,classifier,classifier_hp_name)

def get_input_name(input_feature:Dict):
    numerical_cols = input_feature.get("numerical", []) + input_feature.get("embeddings", [])
    categorical_cols = input_feature.get("categorical", [])
    feature_name = "_".join(numerical_cols + categorical_cols)
    return feature_name

def fold_to_indices(fold_frame: pl.DataFrame, fold_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts indices for training, validation, and test sets based on fold configuration.

    Parameters:
    - fold_frame: DataFrame with columns ['cv_index', '<fold_name>'] where '<fold_name>' column contains
      identifiers for train, validation, and test sets.
    - fold_name: The name of the column in fold_frame that contains the fold identifiers.

    Returns:
    - Tuple of numpy arrays: (train_indices, validation_indices, test_indices)
    """
    # Assuming the fold_frame has a column with the name in fold_name that marks rows as 'train', 'val', or 'test'
    train_indices = fold_frame.filter(pl.col(fold_name) == "train").select("cv_index").to_numpy().flatten()
    val_indices = fold_frame.filter(pl.col(fold_name) == "val").select("cv_index").to_numpy().flatten()
    test_indices = fold_frame.filter(pl.col(fold_name) == "test").select("cv_index").to_numpy().flatten()

    return train_indices, val_indices, test_indices


def preprocess_dataset(df: pl.DataFrame, inputs: List[Dict[str, Union[List[str], List[List[str]]]]],target_column:str="target"):
    feature_arrays = {}
    encoders = {}

    for inp in inputs:
        numerical_cols = inp.get("numerical", []) + inp.get("embeddings", [])
        categorical_cols = inp.get("categorical", [])
        feature_name = "_".join(numerical_cols + categorical_cols)

        X_final,_ = get_features(df, numerical_cols, categorical_cols,target_column = target_column)
        
        feature_arrays[feature_name] = X_final
        print(f"Feature array shape for {feature_name}: {X_final.shape}")

    # Assuming 'target' is the label column
    labels = df[target_column].to_numpy()
    return feature_arrays, labels, encoders

def derive_feature_names(inputs: List[Dict[str, Union[List[str], List[List[str]]]]]) -> List[str]:
    feature_names = []
    for inp in inputs:
        numerical_cols = inp.get("numerical", []) + inp.get("embeddings", [])
        categorical_cols = inp.get("categorical", [])
        feature_name = "_".join(numerical_cols + categorical_cols)
        feature_names.append(feature_name)
    return feature_names


def get_features(df:pl.DataFrame,numerical_cols:list[str],categorical_cols:list[str],return_encoder:Optional[bool] = False,cat_encoder:Optional[OneHotEncoder]=None, target_column:str="target") -> Tuple[np.ndarray,np.ndarray]:
        # print(f"Number of samples inside geat_feature: {df.shape[0]}")
        y_final = df[target_column].to_numpy()
        # print(f"Number of samples for the test set inside geat_feature: {y_final.shape[0]}")
        #get train embeddings
        embeddings = []
        if len(numerical_cols)>0:
            for col in numerical_cols:
                embedding_np = np.array(df[col].to_list())
                embeddings.append(embedding_np)
            embeddings_np = np.concatenate(embeddings,axis=1)
        
        # Selecting only the categorical columns and the target
        encoder = None
        if len(categorical_cols)>0:
            X = df.select(pl.col(categorical_cols))
            # One-hot encoding the categorical variables
            if cat_encoder is None:
                encoder = OneHotEncoder()
            X_encoded = encoder.fit_transform(X)
            X_encoded = X_encoded.toarray()
        #case 1 only embeddings
        if len(categorical_cols)==0 and len(numerical_cols)>0:
            X_final = embeddings_np
        #case 2 only categorical
        elif len(categorical_cols)>0 and len(numerical_cols)==0:
            X_final = X_encoded
        #case 3 both
        elif len(categorical_cols)>0 and len(numerical_cols)>0:
            X_final = np.concatenate([embeddings_np, X_encoded], axis=1)
        else:
            raise ValueError("No features selected")
        if return_encoder:
           
            return X_final,y_final,encoder
        return X_final,y_final

def fit_clf(X_train, y_train, X_val, y_val,X_test, y_test,fold_frame:pl.DataFrame,classifier:str="RandomForest",classifier_hp:dict={},input_features_name:str="") -> Tuple[pl.DataFrame,pl.DataFrame]:
    start_time = time.time()
    # Initialize the Random Forest classifier
    if classifier=="RandomForest":
        if classifier_hp is None:
            classifier_hp = {"random_state":777,"n_estimators":100,"max_depth":5,"n_jobs":-1}
        clf = RandomForestClassifier(**classifier_hp)
        
    elif classifier == "NearestNeighbors":
        if classifier_hp is None:
            classifier_hp = {"n_neighbors":7}
        clf = KNeighborsClassifier(**classifier_hp)
    elif classifier == "MLP":
        if classifier_hp is None:
            classifier_hp = {"alpha":1, "max_iter":1000, "random_state":42, "hidden_layer_sizes":(1000, 500)}
        clf = MLPClassifier(**classifier_hp)
    #create the classifier_hp_name from the dictionary
    classifier_hp_name = "_".join([f"{key}_{value}" for key,value in classifier_hp.items()])
    fold_name = fold_frame.columns[1]
    pred_column_name = "{}_{}_{}_{}_y_pred".format(fold_name,input_features_name,classifier,classifier_hp_name)
    clf = make_pipeline(StandardScaler(), clf)
    train_index_series = fold_frame.filter(pl.col(fold_name)=="train")["cv_index"]
    val_index_series = fold_frame.filter(pl.col(fold_name)=="val")["cv_index"]
    test_index_series = fold_frame.filter(pl.col(fold_name)=="test")["cv_index"]

    # Train the classifier using the training set
    start_train_time = time.time()
    clf.fit(X_train, y_train)
    # Predict on all the folds
    end_train_time = time.time()
    human_readable_train_time = time.strftime("%H:%M:%S", time.gmtime(end_train_time-start_train_time))
    start_pred_time = time.time()
    y_pred_train = clf.predict(X_train)
    y_pred_val = clf.predict(X_val)
    y_pred_test = clf.predict(X_test)
   

    pred_train_df = pl.DataFrame({"cv_index":train_index_series,pred_column_name:y_pred_train})
    pred_val_df = pl.DataFrame({"cv_index":val_index_series,pred_column_name:y_pred_val})
    pred_test_df = pl.DataFrame({"cv_index":test_index_series,pred_column_name:y_pred_test})
    pred_df = pred_train_df.vstack(pred_val_df).vstack(pred_test_df)

    end_pred_time = time.time()
    human_readable_pred_time = time.strftime("%H:%M:%S", time.gmtime(end_pred_time-start_pred_time))

    
    start_eval_time = time.time()
    # Evaluate the classifier
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    mcc_train = matthews_corrcoef(y_train, y_pred_train)
    mcc_test = matthews_corrcoef(y_test, y_pred_test)
    mcc_val = matthews_corrcoef(y_val, y_pred_val)
    end_eval_time = time.time()
    human_readable_eval_time = time.strftime("%H:%M:%S", time.gmtime(end_eval_time-start_eval_time))
    end_time = time.time()
    human_readable_total_time = time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))
    time_dict = {"train_time":human_readable_train_time,
                 "pred_time":human_readable_pred_time,
                 "eval_time":human_readable_eval_time,"total_cls_time":human_readable_total_time}

    results_df = pl.DataFrame({"classifier":[classifier],
                               "classifier_hp":[classifier_hp_name],
                               "fold_name":[fold_name],
                               "pred_name":[pred_column_name],
                               "input_features_name":[input_features_name],
                               "accuracy_train":[accuracy_train],
                                "accuracy_val":[accuracy_val],
                                "accuracy_test":[accuracy_test],
                                "mcc_train":[mcc_train],
                                "mcc_val":[mcc_val],
                                "mcc_test":[mcc_test],
                                "train_index":[train_index_series],
                                "val_index":[val_index_series],
                                "test_index":[test_index_series],
                                "time":[time_dict],
                                }).unnest("time")
    # print nicely formatted: accuracy, cv_scores.mean(), cv_scores.std(), 
    # print(f"Accuracy Test: {accuracy_test}")
    # print(f"Accuracy Val: {accuracy_val}")
    # print(f"MCC Test: {mcc_test}")
    # print(f"MCC Val: {mcc_val}")
    # print(f"Total CLS time: {human_readable_total_time}")
    return pred_df,results_df

def fit_clf_from_np(X_train, y_train, X_val, y_val, X_test, y_test,fold_metadata:dict,
            classifier: str = "RandomForest", classifier_hp: dict = {}, input_features_name: str = "") -> Tuple[pl.DataFrame, pl.DataFrame]:
    start_time = time.time()
    clf = None
    #create classifiers 
    if classifier == "RandomForest":
        clf = RandomForestClassifier(**(classifier_hp or {"random_state": 777, "n_estimators": 100, "max_depth": 5, "n_jobs": -1}))
    elif classifier == "NearestNeighbors":
        clf = KNeighborsClassifier(**(classifier_hp or {"n_neighbors": 7}))
    elif classifier == "MLP":
        clf = MLPClassifier(**(classifier_hp or {"alpha": 1, "max_iter": 1000, "random_state": 42, "hidden_layer_sizes": (1000, 500)}))
    else:
        raise ValueError("Classifier not supported")
    
    #create names 
    classifier_hp_name = get_hp_classifier_name(classifier_hp)
    fold_name = fold_metadata["fold_name"]
    # pred_column_name = "{}_{}_{}_{}_y_pred".format(fold_name,input_features_name,classifier,classifier_hp_name)
    pred_column_name = get_pred_column_name(fold_name,input_features_name,classifier,classifier_hp_name)
    # Train the classifier using the training set
    start_train_time = time.time()
    clf = make_pipeline(StandardScaler(), clf)
    # print("before training")
    # print("X_train shape: ", X_train.shape)
    # print("y_train shape: ", y_train.shape)
    clf.fit(X_train, y_train)
    end_train_time = time.time()
    human_readable_train_time = time.strftime("%H:%M:%S", time.gmtime(end_train_time-start_train_time))

    # Predictions
    start_pred_time = time.time()
    # print("before clf predict")
    y_pred_train = clf.predict(X_train)
    y_pred_val = clf.predict(X_val)
    y_pred_test = clf.predict(X_test)
    end_pred_time = time.time()
    human_readable_pred_time = time.strftime("%H:%M:%S", time.gmtime(end_pred_time-start_pred_time))

    # Evaluation
    start_eval_time = time.time()
    # print("before evaluation")
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    mcc_train = matthews_corrcoef(y_train, y_pred_train) 
    mcc_val = matthews_corrcoef(y_val, y_pred_val) 
    mcc_test = matthews_corrcoef(y_test, y_pred_test) 
    end_eval_time = time.time()
    human_readable_eval_time = time.strftime("%H:%M:%S", time.gmtime(end_eval_time-start_eval_time))
    end_time = time.time()
    human_readable_total_time = time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))

    #package predictions
    pred_train_df = pl.DataFrame({"cv_index":fold_metadata["train_index"],pred_column_name:y_pred_train})
    pred_val_df = pl.DataFrame({"cv_index":fold_metadata["val_index"],pred_column_name:y_pred_val})
    pred_test_df = pl.DataFrame({"cv_index":fold_metadata["test_index"],pred_column_name:y_pred_test})
    pred_df = pred_train_df.vstack(pred_val_df).vstack(pred_test_df)

    #package results

    time_dict = {"train_time":human_readable_train_time,
                "pred_time":human_readable_pred_time,
                "eval_time":human_readable_eval_time,"total_cls_time":human_readable_total_time}

    results_df = pl.DataFrame({"classifier":[classifier],
                               "classifier_hp":[classifier_hp_name],
                               "fold_name":[fold_name],
                               "pred_name":[pred_column_name],
                               "input_features_name":[input_features_name],
                               "accuracy_train":[accuracy_train],
                                "accuracy_val":[accuracy_val],
                                "accuracy_test":[accuracy_test],
                                "mcc_train":[mcc_train],
                                "mcc_val":[mcc_val],
                                "mcc_test":[mcc_test],
                                "train_index":[fold_metadata["train_index"]],
                                "val_index":[fold_metadata["val_index"]],
                                "test_index":[fold_metadata["test_index"]],
                                "time":[time_dict],
                                }).unnest("time")
    

    return pred_df, results_df



def fit_models_modal(models: Dict[str, List[Dict[str, Any]]], feature_name: str, indices_train:np.ndarray,indices_val: np.ndarray,indices_test:np.ndarray,
               fold_meta: Dict[str, Any],mount_directory:str) -> Tuple[List[pl.DataFrame], List[pl.DataFrame]]:
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score,  matthews_corrcoef
    from sklearn.preprocessing import OneHotEncoder
    import time
    def load_arrays_from_mount_modal(feature_name:str):
        X = np.load(os.path.join("/root/cynde_mount",feature_name+".npy"))
        y = np.load(os.path.join("/root/cynde_mount","labels.npy"))
        return X,y
    def fit_clf_from_np_modal(X_train, y_train, X_val, y_val, X_test, y_test,fold_metadata:dict,
            classifier: str = "RandomForest", classifier_hp: dict = {}, input_features_name: str = "") -> Tuple[pl.DataFrame, pl.DataFrame]:
    
        start_time = time.time()
        clf = None
        #create classifiers 
        if classifier == "RandomForest":
            clf = RandomForestClassifier(**(classifier_hp or {"random_state": 777, "n_estimators": 100, "max_depth": 5, "n_jobs": -1}))
        elif classifier == "NearestNeighbors":
            clf = KNeighborsClassifier(**(classifier_hp or {"n_neighbors": 7}))
        elif classifier == "MLP":
            clf = MLPClassifier(**(classifier_hp or {"alpha": 1, "max_iter": 1000, "random_state": 42, "hidden_layer_sizes": (1000, 500)}))
        else:
            raise ValueError("Classifier not supported")
        
        #create names 
        classifier_hp_name = "_".join([f"{key}_{value}" for key,value in classifier_hp.items()])
        fold_name = fold_metadata["fold_name"]
        pred_column_name = "{}_{}_{}_{}_y_pred".format(fold_name,input_features_name,classifier,classifier_hp_name)

        # Train the classifier using the training set
        start_train_time = time.time()
        clf = make_pipeline(StandardScaler(), clf)
        # print("before training")
        # print("X_train shape: ", X_train.shape)
        # print("y_train shape: ", y_train.shape)
        clf.fit(X_train, y_train)
        end_train_time = time.time()
        human_readable_train_time = time.strftime("%H:%M:%S", time.gmtime(end_train_time-start_train_time))

        # Predictions
        start_pred_time = time.time()
        # print("before clf predict")
        y_pred_train = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)
        y_pred_test = clf.predict(X_test)
        end_pred_time = time.time()
        human_readable_pred_time = time.strftime("%H:%M:%S", time.gmtime(end_pred_time-start_pred_time))

        # Evaluation
        start_eval_time = time.time()
        # print("before evaluation")
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_val = accuracy_score(y_val, y_pred_val)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        mcc_train = matthews_corrcoef(y_train, y_pred_train) 
        mcc_val = matthews_corrcoef(y_val, y_pred_val) 
        mcc_test = matthews_corrcoef(y_test, y_pred_test) 
        end_eval_time = time.time()
        human_readable_eval_time = time.strftime("%H:%M:%S", time.gmtime(end_eval_time-start_eval_time))
        end_time = time.time()
        human_readable_total_time = time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))

        #package predictions
        pred_train_df = pl.DataFrame({"cv_index":fold_metadata["train_index"],pred_column_name:y_pred_train})
        pred_val_df = pl.DataFrame({"cv_index":fold_metadata["val_index"],pred_column_name:y_pred_val})
        pred_test_df = pl.DataFrame({"cv_index":fold_metadata["test_index"],pred_column_name:y_pred_test})
        pred_df = pred_train_df.vstack(pred_val_df).vstack(pred_test_df)

        #package results

        time_dict = {"train_time":human_readable_train_time,
                    "pred_time":human_readable_pred_time,
                    "eval_time":human_readable_eval_time,"total_cls_time":human_readable_total_time}

        results_df = pl.DataFrame({"classifier":[classifier],
                                "classifier_hp":[classifier_hp_name],
                                "fold_name":[fold_name],
                                "pred_name":[pred_column_name],
                                "input_features_name":[input_features_name],
                                "accuracy_train":[accuracy_train],
                                    "accuracy_val":[accuracy_val],
                                    "accuracy_test":[accuracy_test],
                                    "mcc_train":[mcc_train],
                                    "mcc_val":[mcc_val],
                                    "mcc_test":[mcc_test],
                                    "train_index":[fold_metadata["train_index"]],
                                    "val_index":[fold_metadata["val_index"]],
                                    "test_index":[fold_metadata["test_index"]],
                                    "time":[time_dict],
                                    }).unnest("time")
        

        return pred_df, results_df
    

def load_arrays_from_mount_modal(feature_name:str):
        X = np.load(os.path.join("/root/cynde_mount",feature_name+".npy"))
        y = np.load(os.path.join("/root/cynde_mount","labels.npy"))
        return X,y


def fit_clf_from_np_modal(X_train, y_train, X_val, y_val, X_test, y_test,fold_metadata:dict,
        classifier: str = "RandomForest", classifier_hp: dict = {}, input_features_name: str = "") -> Tuple[pl.DataFrame, pl.DataFrame]:

    start_time = time.time()
    clf = None
    #create classifiers 
    if classifier == "RandomForest":
        clf = RandomForestClassifier(**(classifier_hp or {"random_state": 777, "n_estimators": 100, "max_depth": 5, "n_jobs": -1}))
    elif classifier == "NearestNeighbors":
        clf = KNeighborsClassifier(**(classifier_hp or {"n_neighbors": 7}))
    elif classifier == "MLP":
        clf = MLPClassifier(**(classifier_hp or {"alpha": 1, "max_iter": 1000, "random_state": 42, "hidden_layer_sizes": (1000, 500)}))
    else:
        raise ValueError("Classifier not supported")
    
    #create names 
    classifier_hp_name = "_".join([f"{key}_{value}" for key,value in classifier_hp.items()])
    classifier_hp_name = get_hp_classifier_name(classifier_hp)
    fold_name = fold_metadata["fold_name"]
    pred_column_name = "{}_{}_{}_{}_y_pred".format(fold_name,input_features_name,classifier,classifier_hp_name)

    # Train the classifier using the training set
    start_train_time = time.time()
    clf = make_pipeline(StandardScaler(), clf)
    # print("before training")
    # print("X_train shape: ", X_train.shape)
    # print("y_train shape: ", y_train.shape)
    clf.fit(X_train, y_train)
    end_train_time = time.time()
    human_readable_train_time = time.strftime("%H:%M:%S", time.gmtime(end_train_time-start_train_time))

    # Predictions
    start_pred_time = time.time()
    # print("before clf predict")
    y_pred_train = clf.predict(X_train)
    print("prediction type: ", type(y_pred_train))
    #check val length before prediction, it could be empty
    if len(X_val)>0:
        y_pred_val = clf.predict(X_val)
    else:
        y_pred_val = []
    y_pred_test = clf.predict(X_test)
    end_pred_time = time.time()
    human_readable_pred_time = time.strftime("%H:%M:%S", time.gmtime(end_pred_time-start_pred_time))

    # Evaluation
    start_eval_time = time.time()
    # print("before evaluation")
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    mcc_train = matthews_corrcoef(y_train, y_pred_train) 
    mcc_val = matthews_corrcoef(y_val, y_pred_val) 
    mcc_test = matthews_corrcoef(y_test, y_pred_test) 
    end_eval_time = time.time()
    human_readable_eval_time = time.strftime("%H:%M:%S", time.gmtime(end_eval_time-start_eval_time))
    end_time = time.time()
    human_readable_total_time = time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))

    #package predictions
    pred_train_df = pl.DataFrame({"cv_index":fold_metadata["train_index"],pred_column_name:y_pred_train})
    pred_val_df = pl.DataFrame({"cv_index":fold_metadata["val_index"],pred_column_name:y_pred_val})
    pred_test_df = pl.DataFrame({"cv_index":fold_metadata["test_index"],pred_column_name:y_pred_test})
    pred_df = pred_train_df.vstack(pred_val_df).vstack(pred_test_df)

    #package results

    time_dict = {"train_time":human_readable_train_time,
                "pred_time":human_readable_pred_time,
                "eval_time":human_readable_eval_time,"total_cls_time":human_readable_total_time}

    results_df = pl.DataFrame({"classifier":[classifier],
                            "classifier_hp":[classifier_hp_name],
                            "fold_name":[fold_name],
                            "pred_name":[pred_column_name],
                            "input_features_name":[input_features_name],
                            "accuracy_train":[accuracy_train],
                                "accuracy_val":[accuracy_val],
                                "accuracy_test":[accuracy_test],
                                "mcc_train":[mcc_train],
                                "mcc_val":[mcc_val],
                                "mcc_test":[mcc_test],
                                "train_index":[fold_metadata["train_index"]],
                                "val_index":[fold_metadata["val_index"]],
                                "test_index":[fold_metadata["test_index"]],
                                "time":[time_dict],
                                }).unnest("time")
    

    return pred_df, results_df

---

## cv

import polars as pl
import numpy as np
from typing import List, Tuple, Union, Dict, Any, Generator, Optional
import time
from cynde.functional.classify import get_features, fit_clf,fit_clf_from_np, fold_to_indices, preprocess_dataset
import os
import math

def shuffle_frame(df:pl.DataFrame):
    return df.sample(fraction=1,shuffle=True)

def slice_frame(df:pl.DataFrame, num_slices:int, shuffle:bool = False, explode:bool = False):
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
def check_add_cv_index(df:pl.DataFrame):
    if "cv_index" not in df.columns:
        df = df.with_row_index(name="cv_index")
    return df
def vanilla_kfold(df:pl.DataFrame,group,k:int,shuffle:bool=True,pre_name:str="",target_names:Tuple[str,str] = ("train","test")):
    #we will use the row index to split the data
    #first we will shuffle the data
    df = check_add_cv_index(df)
    if shuffle:
        df = shuffle_frame(df)
    #then we will split the data into k slices
    df_slices = slice_frame(df,k,shuffle=False,explode=False)
    index_df = df.select(pl.col("cv_index"))
    for i in range(k):
        test_set = df_slices[i]
        train_set = [df_slices[j] for j in range(k) if j!=i]
        train_set = pl.concat(train_set)

        test_df = test_set.with_columns([pl.lit(target_names[1]).alias(pre_name+"fold_{}".format(i))])
        train_df = train_set.with_columns([pl.lit(target_names[0]).alias(pre_name+"fold_{}".format(i))])
        fold_df = train_df.vstack(test_df)
        index_df = index_df.join(fold_df, on="cv_index", how="left")
    return index_df

def purged_kfold(df:pl.DataFrame,group:List[str],k:int,shuffle:bool=True,pre_name:str="",target_names:Tuple[str,str] = ("train","test")):
    #group is a list of columns that will be used to group the data
    #k is the number of splits
    #we will use the group to split the data into k groups and then we will use the group to make sure that the same group is not in the train and test set
    #we will use the row index to split the data
    #first we will shuffle the data
    df = check_add_cv_index(df)
    gdf = df.group_by(group).agg(pl.col("cv_index"))
    #then we will split the data into k slices we need to explode the index since we are working on groups
    gdf_slices = slice_frame(gdf,k,shuffle=shuffle,explode=True)
    #then we will iterate over the slices and use the slice as the test set and the rest as the train set
    index_df = df.select(pl.col("cv_index"))
    for i in range(k):
        test_set = gdf_slices[i]
        train_set = [gdf_slices[j] for j in range(k) if j!=i]
        train_set = pl.concat(train_set)

        test_df = test_set.with_columns([pl.lit(target_names[1]).alias(pre_name+"fold_{}".format(i))])
        train_df = train_set.with_columns([pl.lit(target_names[0]).alias(pre_name+"fold_{}".format(i))])
        fold_df = train_df.vstack(test_df)
        index_df = index_df.join(fold_df, on="cv_index", how="left")
    return index_df

def stratified_kfold(df:pl.DataFrame,group:List[str],k:int,shuffle:bool=True,pre_name:str="",target_names:Tuple[str,str] = ("train","test")):
    df = check_add_cv_index(df)
    sdf = df.group_by(group).agg(pl.col("cv_index"))
    if shuffle:
        sdf = sdf.with_columns(pl.col("cv_index").list.sample(fraction=1,shuffle=True))
    sliced = sdf.select(pl.col("cv_index").map_elements(lambda s: hacky_list_relative_slice(s,k)).alias("hacky_cv_index")).unnest("hacky_cv_index")

    index_df = df.select(pl.col("cv_index"))

    for i in range(k):
            test_set = sliced.select(pl.col("fold_{}".format(i)).alias("cv_index")).explode("cv_index")
            train_set = sliced.select(pl.concat_list([sliced["fold_{}".format(j)] for j in range(k) if j!=i]).alias("cv_index")).explode("cv_index")
            test_df = test_set.with_columns([pl.lit(target_names[1]).alias(pre_name+"fold_{}".format(i))])
            train_df = train_set.with_columns([pl.lit(target_names[0]).alias(pre_name+"fold_{}".format(i))])
            index_df = index_df.join(train_df.vstack(test_df), on="cv_index", how="left")
    return index_df

def divide_df_by_target(df: pl.DataFrame,target:str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    if target not in df.columns:
        raise ValueError(f"Target {target} not in columns")
    print(f"Total rows: {df.shape[0]}")
    target_rows = df.filter(pl.col(target)==True)
    other = df.filter(pl.col(target)==False)
    print(f"Target {target} rows: {target_rows.shape[0]}")
    print(f"Other rows: {other.shape[0]}")
    return target_rows, other

def resample_dataset(df:pl.DataFrame,df_target: pl.DataFrame, df_other:pl.DataFrame, size: int = math.inf) -> pl.DataFrame:
    size_target,size_other = [df_target.shape[0], df_other.shape[0]]
    min_size = min(size_target, size_other,size)
    df_index = sample_dataset_index(df_target, df_other, min_size)
    return df.filter(pl.col("cv_index").is_in(df_index["cv_index"]))

def sample_dataset_index(df_target: pl.DataFrame, df_other:pl.DataFrame, size: int = math.inf) -> pl.DataFrame:
    size_target,size_other = [df_target.shape[0], df_other.shape[0]]
    min_size = min(size_target, size_other,size)
    return df_target.select(pl.col("cv_index")).sample(min_size).vstack(df_other.select(pl.col("cv_index")).sample(min_size))

def vanilla_resample_kfold(df:pl.DataFrame,group:List[str],k:int,shuffle:bool=True,pre_name:str="",target_names:Tuple[str,str] = ("train","test"), size:int = math.inf):
    target = group[0]
    df =check_add_cv_index(df)
    df_only_target = df.select(pl.col(["cv_index",target]))
    df_target, df_other = divide_df_by_target(df_only_target, target)
    df_resampled = resample_dataset(df_only_target, df_target, df_other, size)
    if pre_name == "":
        pre_name = f"resampled_by_{target}_"
    index_df = stratified_kfold(df_resampled, group,k,shuffle=shuffle,pre_name=pre_name,target_names=target_names)
    return index_df
        

def get_cv_function(cv_type:str):
    print(f"cv_type: {cv_type}")
    if cv_type == "purged":
        return purged_kfold
    elif cv_type == "stratified":
        return stratified_kfold
    elif cv_type == "vanilla":
        return vanilla_kfold
    elif cv_type == "resample":
        return vanilla_resample_kfold
    else:
        raise ValueError("cv_type can only be purged, stratified or vanilla")

def validate_cv_type(cv_type: Tuple[str,str]):
    for cv_subtype in cv_type:
        if cv_subtype not in ["purged","stratified","vanilla","resample"]:
            raise ValueError("cv_type can only be purged, stratified or vanilla")
    return cv_type[0],cv_type[1]

def derive_cv_columns(group_outer:List[str],group_inner:List[str]):
    cv_columns = ["cv_index"]
    if group_outer is not None:
        name_group_outer = "_".join(group_outer)
        cv_columns += group_outer
    else:
        name_group_outer = ""
    if group_inner is not None:
        name_group_inner = "_".join(group_inner)
        cv_columns += group_inner
    else:
        name_group_inner = ""
    cv_columns = list(set(cv_columns))
    return cv_columns,name_group_outer,name_group_inner


def nested_cv(df:pl.DataFrame, cv_type: Tuple[str,str], group_outer:List[str],k_outer:int,group_inner:List[str],k_inner:int,r_outer:int =1, r_inner:int =1,return_joined : bool = False):
    if "precomputed" in cv_type:
        print("detected precomputed cv")
        df = check_add_cv_index(df)
        if return_joined:        
            return df
        else:
            folds_columns = [col for col in  df.columns if "fold" in col]
            return df.select(pl.col(["cv_index"]+folds_columns))

    
    outer_type,inner_type = validate_cv_type(cv_type)
        
    df = check_add_cv_index(df)
    
    cv_columns,name_group_outer,name_group_inner = derive_cv_columns(group_outer,group_inner)

    cv_df = df.select(pl.col(cv_columns))
    
    outer_cv_function = get_cv_function(outer_type)
    inner_cv_function = get_cv_function(inner_type)

    for r_out in range(r_outer):

        outer_pre_name = "outer_{}_{}_replica_{}_".format(outer_type,name_group_outer,r_out)
        outer_folds = outer_cv_function(df,group_outer,k_outer,shuffle=True,pre_name=outer_pre_name,target_names=("dev","test"))
        cv_df = cv_df.join(outer_folds, on="cv_index", how="left")

        for k_out in range(k_outer):
            for r_in in range(r_inner):

                inner_pre_name = "inner_{}_{}_replica_{}_".format(inner_type,name_group_inner,r_in)
                target_column_name = "{}fold_{}".format(outer_pre_name,k_out)
                dev_df = cv_df.filter(pl.col(target_column_name)=="dev")
                test_df = cv_df.select(pl.col(["cv_index",target_column_name])).filter(pl.col(target_column_name)=="test")
                complete_pre_name = "{}fold_{}_{}".format(outer_pre_name,k_out,inner_pre_name)
                inner_folds = inner_cv_function(dev_df,group_inner,k_inner,shuffle=True,pre_name=complete_pre_name,target_names=("train","val"))
                col_df_list= [test_df.select(pl.col("cv_index"))]
                for col in inner_folds.columns:
                    if col != "cv_index":
                        col_df = test_df.select(pl.col(target_column_name).alias(col))
                        col_df_list.append(col_df)
                test_rows_df =pl.concat(col_df_list,how="horizontal")
                inner_folds = inner_folds.vstack(test_rows_df)
        
                
                cv_df = cv_df.join(inner_folds, on="cv_index", how="left")
    if return_joined:
        return df.join(cv_df, on=cv_columns, how="left")
    else:
        return cv_df.sort("cv_index")
    
def get_fold_name_cv(group_outer:List[str],
                    cv_type: Tuple[str,str],
                    r_outer:int,
                    k_outer:int,
                    group_inner:List[str],
                    r_inner:int,
                    k_inner:int):
    if "precomputed" in cv_type:
        return "fold_{}".format(k_inner+1)
    return "outer_{}_{}_replica_{}_fold_{}_inner_{}_{}_replica_{}_fold_{}".format(cv_type[0],
                                                                                  "_".join(group_outer),   
                                                                                   r_outer,
                                                                                   k_outer,
                                                                                   cv_type[1],
                                                                                   "_".join(group_inner),
                                                                                   r_inner,
                                                                                   k_inner)

RESULTS_SCHEMA = {"classifier":pl.Utf8,
                  "classifier_hp":pl.Utf8,
                "fold_name":pl.Utf8,
                "pred_name":pl.Utf8,
                "input_features_name":pl.Utf8,
                "accuracy_train":pl.Float64,
                "accuracy_val":pl.Float64,
                "accuracy_test":pl.Float64,
                "mcc_train":pl.Float64,
                "mcc_val":pl.Float64,
                "mcc_test":pl.Float64,
                "train_index":pl.List(pl.UInt32),
                "val_index":pl.List(pl.UInt32),
                "test_index":pl.List(pl.UInt32),
                "train_time":pl.Utf8,
                "pred_time":pl.Utf8,
                "eval_time":pl.Utf8,
                "total_cls_time":pl.Utf8,
                "k_outer":pl.Int64,
                "k_inner":pl.Int64,
                "r_outer":pl.Int64,
                "r_inner":pl.Int64}

def fold_to_dfs(df:pl.DataFrame,fold_name:str) -> Tuple[pl.DataFrame,pl.DataFrame,pl.DataFrame]:
    df_train = df.filter(pl.col(fold_name)=="train")
    df_val = df.filter(pl.col(fold_name)=="val")
    df_test = df.filter(pl.col(fold_name)=="test")
    return df_train,df_val,df_test

def train_nested_cv(df:pl.DataFrame,
                    cv_type: Tuple[str,str],
                    inputs:List[Dict[str,Union[List[str],List[List[str]]]]],
                    models:Dict[str,List[Dict[str,Any]]],
                    group_outer:List[str],
                    k_outer:int,
                    group_inner:List[str],
                    k_inner:int,
                    r_outer:int =1,
                    r_inner:int =1,
                    save_name:str="nested_cv_out",
                    base_path:Optional[str]=None) -> Tuple[pl.DataFrame,pl.DataFrame]:
    
    df = check_add_cv_index(df)
    pred_df = nested_cv(df, cv_type, group_outer, k_outer, group_inner, k_inner, r_outer, r_inner, return_joined=False)
    print(pred_df.columns)
    cv_df = df.join(pred_df, on="cv_index", how="left")
    results_df = pl.DataFrame(schema = RESULTS_SCHEMA)

    for r_o in range(r_outer):
        for k_o in range(k_outer):
            for r_i in range(r_inner):
                for k_i in range(k_inner):
                    fold_name = get_fold_name_cv(group_outer,cv_type,r_o,k_o,group_inner,r_i,k_i)
                    fold_frame = cv_df.select(pl.col("cv_index"),pl.col(fold_name))
                    df_train,df_val,df_test = fold_to_dfs(cv_df,fold_name)
                    for inp in inputs:
                        #numerical features are inp["numerical"] and inp["embeddings"] are the embeddings
                        numerical = inp.get("numerical",[])+inp.get("embeddings",[])
                        categorical = inp.get("categorical",[])
                        feature_name = "_".join(numerical+categorical)
                        x_tr, y_tr,cat_encoder = get_features(df_train,numerical,categorical,return_encoder=True)
                        x_val, y_val = get_features(df_val,numerical,categorical,cat_encoder=cat_encoder)
                        x_te, y_te = get_features(df_test,numerical,categorical,cat_encoder=cat_encoder)

                        for model,hp_list in models.items():
                            for hp in hp_list:
                                pred_df_col,results_df_row = fit_clf(x_tr, y_tr, x_val, y_val,x_te, y_te,
                                                                     classifier=model,
                                                                     classifier_hp= hp,
                                                                     fold_frame=fold_frame,
                                                                     input_features_name=feature_name)
                                
                                fold_meta_data_df = pl.DataFrame({"k_outer":[k_o],"k_inner":[k_i],"r_outer":[r_o],"r_inner":[r_i]})
                                results_df_row = pl.concat([results_df_row,fold_meta_data_df],how="horizontal")                   
                                results_df = results_df.vstack(results_df_row)

                                pred_df = pred_df.join(pred_df_col, on="cv_index", how="left")
    time_stamp_str_url_compatible = time.strftime("%Y-%m-%d_%H-%M-%S")
    #results are frame num_models and pred_df is a frame num_sample 
    #joined df is 
    save_name_res = "results_"+time_stamp_str_url_compatible+"_"+save_name+".parquet"
    save_name_pred = "predictions_"+time_stamp_str_url_compatible+"_"+save_name+".parquet"
    save_name_joined_df = "joined_"+time_stamp_str_url_compatible+"_"+save_name+".parquet"
    #use os to merge filenames with data_processed folder
    #goes up one directory and then into data_processed
    #retrive the path of teh cynde package
    if base_path is not None:
        up_one = os.path.join(base_path,"data_processed")
    else:
        up_one = os.path.join(os.getcwd(),"data_processed")
    save_name_res = os.path.join(up_one,save_name_res)
    save_name_pred = os.path.join(up_one,save_name_pred)
    save_name_joined_df = os.path.join(up_one,save_name_joined_df)
    print(f"Saving results to {save_name_res}")
    results_df.write_parquet(save_name_res)
    pred_df.write_parquet(save_name_pred)
    df.join(pred_df, on="cv_index", how="left").write_parquet(save_name_joined_df)
    return results_df,pred_df



def generate_folds(cv_df: pl.DataFrame, cv_type: Tuple[str, str], inputs: List[Dict[str, Union[List[str], List[List[str]]]]],
                   group_outer: List[str], k_outer: int, group_inner: List[str], k_inner: int, r_outer: int, r_inner: int) -> Generator:
    for r_o in range(r_outer):
        for k_o in range(k_outer):
            for r_i in range(r_inner):
                for k_i in range(k_inner):
                    fold_name = get_fold_name_cv(group_outer, cv_type, r_o, k_o, group_inner, r_i, k_i)
                    fold_frame = cv_df.select(pl.col("cv_index"), pl.col(fold_name))
                    df_train, df_val, df_test = fold_to_dfs(cv_df, fold_name)
                    for inp in inputs:
                        numerical = inp.get("numerical", []) + inp.get("embeddings", [])
                        categorical = inp.get("categorical", [])
                        feature_name = "_".join(numerical + categorical)
                        x_tr, y_tr, cat_encoder = get_features(df_train, numerical, categorical, return_encoder=True)
                        x_val, y_val = get_features(df_val, numerical, categorical, cat_encoder=cat_encoder)
                        x_te, y_te = get_features(df_test, numerical, categorical, cat_encoder=cat_encoder)

                        fold_meta = {
                            "r_outer": r_o,
                            "k_outer": k_o,
                            "r_inner": r_i,
                            "k_inner": k_i,
                            "embeddings": inp.get("embeddings", [])
                        }
                        
                        yield fold_name, fold_frame, feature_name, x_tr, y_tr, x_val, y_val, x_te, y_te, fold_meta


def save_results(df:pl.DataFrame,results_df: pl.DataFrame, pred_df: pl.DataFrame, save_name: str, base_path:Optional[str]=None):
    time_stamp_str_url_compatible = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_name_res = f"results_{time_stamp_str_url_compatible}_{save_name}.parquet"
    save_name_pred = f"predictions_{time_stamp_str_url_compatible}_{save_name}.parquet"
    save_name_joined_df = f"joined_{time_stamp_str_url_compatible}_{save_name}.parquet"
    if base_path is not None:
        up_one = os.path.join(base_path, "data_processed")
 
    else:
        up_one = os.path.join(os.getcwd(), "data_processed")
    if not os.path.exists(up_one):
        os.mkdir(up_one)
    save_name_res = os.path.join(up_one, save_name_res)
    save_name_pred = os.path.join(up_one, save_name_pred)
    save_name_joined_df = os.path.join(up_one, save_name_joined_df)
    print(f"Saving results to {save_name_res}")
    results_df.write_parquet(save_name_res)
    pred_df.write_parquet(save_name_pred)
    df.join(pred_df, on="cv_index", how="left").write_parquet(save_name_joined_df)

def fit_models(models: Dict[str, List[Dict[str, Any]]], fold_frame: pl.DataFrame, feature_name: str,
               x_tr: np.ndarray, y_tr: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
               fold_meta: Dict[str, Any]) -> Tuple[List[pl.DataFrame], List[pl.DataFrame]]:
    pred_list = []
    results_list = []
    for model, hp_list in models.items():
        for hp in hp_list:
            pred_df_col, results_df_row = fit_clf(x_tr, y_tr, x_val, y_val, x_te, y_te,
                                                  classifier=model,
                                                  classifier_hp=hp,
                                                  fold_frame=fold_frame,
                                                  input_features_name=feature_name)
            pred_list.append(pred_df_col)
            
            fold_meta_data_df = pl.DataFrame({
                "k_outer": [fold_meta["k_outer"]],
                "k_inner": [fold_meta["k_inner"]],
                "r_outer": [fold_meta["r_outer"]],
                "r_inner": [fold_meta["r_inner"]]
            })
            
            results_df_row = pl.concat([results_df_row, fold_meta_data_df], how="horizontal") 
            results_list.append(results_df_row)
    return pred_list, results_list







def train_nested_cv_simple(df: pl.DataFrame,
                           cv_type: Tuple[str, str],
                           inputs: List[Dict[str, Union[List[str], List[List[str]]]]],
                           models: Dict[str, List[Dict[str, Any]]],
                           group_outer: List[str],
                           k_outer: int,
                           group_inner: List[str],
                           k_inner: int,
                           r_outer: int = 1,
                           r_inner: int = 1,
                           save_name: str = "nested_cv_out",
                           base_path: Optional[str] = None,
                           skip_class: bool = False) -> Tuple[pl.DataFrame, pl.DataFrame]:
    start_time = time.time()
    df = check_add_cv_index(df)
    pred_df = nested_cv(df, cv_type, group_outer, k_outer, group_inner, k_inner, r_outer, r_inner, return_joined=False)
    
    cv_df = df.join(pred_df, on="cv_index", how="left")
    results_df = pl.DataFrame(schema=RESULTS_SCHEMA)
    cv_creation_time = time.time()
    print(f"cv_creation_time: {cv_creation_time - start_time}")
    all_pred_list = []
    all_results_list = []

    for fold_info in generate_folds(cv_df, cv_type, inputs, group_outer, k_outer, group_inner, k_inner, r_outer, r_inner):
        fold_name, fold_frame, feature_name, x_tr, y_tr, x_val, y_val, x_te, y_te, fold_meta = fold_info
        if not skip_class:
            pred_list, results_list = fit_models(models, fold_frame, feature_name, x_tr, y_tr, x_val, y_val, x_te, y_te, fold_meta)
            all_pred_list.extend(pred_list)
            all_results_list.extend(results_list)
    fit_time = time.time()
    print(f"fit_time: {fit_time - cv_creation_time}")
    print(f"average time per fold: {(fit_time - cv_creation_time) / (k_outer * k_inner * r_outer * r_inner)}")

    # Aggregate results
    for results_df_row in all_results_list:
        results_df = results_df.vstack(results_df_row)

    for pred_df_col in all_pred_list:
        pred_df = pred_df.join(pred_df_col, on="cv_index", how="left")
    aggregation_time = time.time()
    print(f"aggregation_time: {aggregation_time - fit_time}")
    save_results(df, results_df, pred_df, save_name, base_path=base_path)
    save_time = time.time()
    print(f"save_time: {save_time - aggregation_time}")
    print(f"total_time: {save_time - start_time}")
    print(f"Total average time per fold: {(save_time - start_time) / (k_outer * k_inner * r_outer * r_inner)}")
    return results_df, pred_df

def generate_folds_from_np(cv_df: pl.DataFrame, cv_type: Tuple[str, str], feature_arrays: Dict[str, np.ndarray],
                    labels: np.ndarray, group_outer: List[str], k_outer: int, group_inner: List[str], k_inner: int,
                    r_outer: int, r_inner: int) -> Generator:
    for r_o in range(r_outer):
        for k_o in range(k_outer):
            for r_i in range(r_inner):
                for k_i in range(k_inner):
                    fold_name = get_fold_name_cv(group_outer, cv_type, r_o, k_o, group_inner, r_i, k_i)
                    indices_train, indices_val, indices_test = fold_to_indices(cv_df, fold_name)

                    # Iterate directly over feature_arrays, as inputs are now preprocessed
                    for feature_name, X in feature_arrays.items():
                        y = labels  # Labels are common for all feature sets
                        # print(f"Feature name: {feature_name}")
                        # print(f"Shapes: {X.shape}, {y.shape}")
                        # print(f"Indices: {indices_train.shape}, {indices_val.shape}, {indices_test.shape}")
                        # Use indices to select the appropriate subsets for the current fold
                        x_tr, y_tr = X[indices_train,:], y[indices_train]
                        x_val, y_val = X[indices_val,:], y[indices_val]
                        x_te, y_te = X[indices_test,:], y[indices_test]

                        fold_meta = {
                            "r_outer": r_o,
                            "k_outer": k_o,
                            "r_inner": r_i,
                            "k_inner": k_i,
                            "fold_name": fold_name,
                            "train_index": pl.Series(indices_train),
                            "val_index": pl.Series(indices_val),
                            "test_index": pl.Series(indices_test),
                        }

                        yield fold_name, feature_name, x_tr, y_tr, x_val, y_val, x_te, y_te, fold_meta

# def generate_folds_from_np_modal_compatible(cv_df: pl.DataFrame, cv_type: Tuple[str, str], feature_arrays: Dict[str, np.ndarray],
#                            labels: np.ndarray, group_outer: List[str], k_outer: int, group_inner: List[str], k_inner: int,
#                            r_outer: int, r_inner: int) -> Generator:
#     for r_o in range(r_outer):
#         for k_o in range(k_outer):
#             for r_i in range(r_inner):
#                 for k_i in range(k_inner):
#                     fold_name = get_fold_name_cv(group_outer, cv_type, r_o, k_o, group_inner, r_i, k_i)
#                     indices_train, indices_val, indices_test = fold_to_indices(cv_df, fold_name)

#                     for feature_name, X in feature_arrays.items():
#                         y = labels
#                         x_tr, y_tr = X[indices_train,:], y[indices_train]
#                         x_val, y_val = X[indices_val,:], y[indices_val]
#                         x_te, y_te = X[indices_test,:], y[indices_test]

#                         fold_meta = {
#                             "r_outer": r_o,
#                             "k_outer": k_o,
#                             "r_inner": r_i,
#                             "k_inner": k_i,
#                             "fold_name": fold_name,
#                             "train_index": pl.Series(indices_train),
#                             "val_index": pl.Series(indices_val),
#                             "test_index": pl.Series(indices_test),
#                         }

#                         yield (x_tr, y_tr, x_val, y_val, x_te, y_te, fold_meta, feature_name)


def fit_models_from_np(models: Dict[str, List[Dict[str, Any]]], feature_name: str,
               x_tr: np.ndarray, y_tr: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
               fold_meta: Dict[str, Any]) -> Tuple[List[pl.DataFrame], List[pl.DataFrame]]:
    pred_list = []
    results_list = []
    for model, hp_list in models.items():
        for hp in hp_list:
            # print("before fit_clfrom_np")
            # print(f"Shapes: {x_tr.shape}, {y_tr.shape}, {x_val.shape}, {y_val.shape}, {x_te.shape}, {y_te.shape}")
            pred_df_col, results_df_row = fit_clf_from_np(x_tr, y_tr, x_val, y_val, x_te, y_te,
                                                        fold_metadata=fold_meta,
                                                        classifier=model,
                                                        classifier_hp=hp,
                                                        input_features_name=feature_name)
            pred_list.append(pred_df_col)
            fold_meta_data_df = pl.DataFrame({
                "k_outer": [fold_meta["k_outer"]],
                "k_inner": [fold_meta["k_inner"]],
                "r_outer": [fold_meta["r_outer"]],
                "r_inner": [fold_meta["r_inner"]]
            })
            # print("schema", fold_meta_data_df.schema)
            # print(f"shape of row before concat: {results_df_row.shape}")
            results_df_row = pl.concat([results_df_row, fold_meta_data_df], how="horizontal") 
            # print(f"shape of row after concat: {results_df_row.shape}")
            
            results_list.append(results_df_row)
    return pred_list, results_list

def train_nested_cv_from_np(df: pl.DataFrame,
                           cv_type: Tuple[str, str],
                           inputs: List[Dict[str, Union[List[str], List[List[str]]]]],
                           models: Dict[str, List[Dict[str, Any]]],
                           group_outer: List[str],
                           k_outer: int,
                           group_inner: List[str],
                           k_inner: int,
                           r_outer: int = 1,
                           r_inner: int = 1,
                           save_name: str = "nested_cv_out",
                           base_path: Optional[str] = None,
                           skip_class: bool = False) -> Tuple[pl.DataFrame, pl.DataFrame]:
    start_time = time.time()
    df = check_add_cv_index(df)
    pred_df = nested_cv(df, cv_type, group_outer, k_outer, group_inner, k_inner, r_outer, r_inner, return_joined=False)
    cv_df = df.join(pred_df, on="cv_index", how="left")
    results_df = pl.DataFrame(schema=RESULTS_SCHEMA)
    print("results schema: ", results_df.schema)
    print(f"results shape: {results_df.shape}")

    # Preprocess the dataset
    preprocess_start_time = time.time()
    feature_arrays, labels, _ = preprocess_dataset(df, inputs)
    preprocess_end_time = time.time()
    print(f"Preprocessing completed in {preprocess_end_time - preprocess_start_time} seconds")

    all_pred_list = []
    all_results_list = []

    # Generate folds and fit models
    folds_generation_start_time = time.time()
    for fold_info in generate_folds_from_np(cv_df, cv_type, feature_arrays, labels, group_outer, k_outer, group_inner, k_inner, r_outer, r_inner):
        fold_name, feature_name, x_tr, y_tr, x_val, y_val, x_te, y_te, fold_meta = fold_info
        if not skip_class:
            
            pred_list, results_list = fit_models_from_np(models,
                                                        feature_name, x_tr, y_tr, x_val, y_val, x_te, y_te, fold_meta)
            all_pred_list.extend(pred_list)
            all_results_list.extend(results_list)
    folds_generation_end_time = time.time()
    print(f"Folds generation and model fitting completed in {folds_generation_end_time - folds_generation_start_time} seconds")
    print(f"Average time per fold: {(folds_generation_end_time - folds_generation_start_time) / (k_outer * k_inner * r_outer * r_inner)}")

    # Aggregate and save results
    aggregation_start_time = time.time()
    for results_df_row in all_results_list:
        results_df = results_df.vstack(results_df_row)

    for pred_df_col in all_pred_list:
        pred_df = pred_df.join(pred_df_col, on="cv_index", how="left")
    aggregation_end_time = time.time()
    print(f"Aggregation of results completed in {aggregation_end_time - aggregation_start_time} seconds")

    save_results_start_time = time.time()
    save_results(df, results_df, pred_df, save_name, base_path=base_path)
    save_results_end_time = time.time()
    print(f"Saving results completed in {save_results_end_time - save_results_start_time} seconds")

    total_end_time = time.time()
    print(f"Total training and processing time: {total_end_time - start_time} seconds")
    print(f"Total average time per fold: {(total_end_time - start_time) / (k_outer * k_inner * r_outer * r_inner)} seconds")

    return results_df, pred_df


def generate_folds_from_np_modal_compatible(models: Dict[str, List[Dict[str, Any]]],cv_df: pl.DataFrame, cv_type: Tuple[str, str], feature_names: List[str],
                           group_outer: List[str], k_outer: int, group_inner: List[str], k_inner: int,
                           r_outer: int, r_inner: int, mount_directory:str) -> Generator:
    for r_o in range(r_outer):
        for k_o in range(k_outer):
            for r_i in range(r_inner):
                for k_i in range(k_inner):
                    fold_name = get_fold_name_cv(group_outer, cv_type, r_o, k_o, group_inner, r_i, k_i)
                    indices_train, indices_val, indices_test = fold_to_indices(cv_df, fold_name)

                    for feature_name in feature_names:
                        # y = labels
                        # x_tr, y_tr = X[indices_train,:], y[indices_train]
                        # x_val, y_val = X[indices_val,:], y[indices_val]
                        # x_te, y_te = X[indices_test,:], y[indices_test]

                        #create an adequate name for each file and save to .npy in the mount dictory

                        fold_meta = {
                            "r_outer": r_o,
                            "k_outer": k_o,
                            "r_inner": r_i,
                            "k_inner": k_i,
                            "fold_name": fold_name,
                            "train_index": pl.Series(indices_train),
                            "val_index": pl.Series(indices_val),
                            "test_index": pl.Series(indices_test),
                        }

                        yield (models, feature_name, indices_train,indices_val,indices_test, fold_meta, mount_directory)


---

## distributed cv

from modal import Image
import modal
import polars as pl
import os
import numpy as np
import polars as pl
import time
import os
from typing import Tuple, Optional, Dict, List, Any, Union
from cynde.functional.cv import generate_folds_from_np_modal_compatible, check_add_cv_index, preprocess_dataset, nested_cv, RESULTS_SCHEMA
from cynde.functional.classify import fit_clf_from_np_modal, load_arrays_from_mount_modal

cv_stub = modal.Stub("distributed_cv")
LOCAL_MOUNT_PATH = os.getenv('MODAL_MOUNT')
    
datascience_image = (
    Image.debian_slim(python_version="3.12.1")
    .apt_install("git")
    .pip_install("polars","scikit-learn","openai","tiktoken", force_build=True)
    
    .run_commands("git clone https://github.com/Neural-Dragon-AI/Cynde/")
    .env({"CYNDE_DIR": "/opt/cynde"})
    .run_commands("cd Cynde && pip install -r requirements.txt && pip install .")
)
with datascience_image.imports():
    import polars as pl
    import sklearn as sk
    import cynde.functional as cf

@cv_stub.function(image=datascience_image, mounts=[modal.Mount.from_local_dir(LOCAL_MOUNT_PATH, remote_path="/root/cynde_mount")])
def fit_models_modal(models: Dict[str, List[Dict[str, Any]]], feature_name: str, indices_train:np.ndarray,indices_val: np.ndarray,indices_test:np.ndarray,
               fold_meta: Dict[str, Any],mount_directory:str) -> Tuple[List[pl.DataFrame], List[pl.DataFrame]]:

    
    pred_list = []
    results_list = []
    X,y = load_arrays_from_mount_modal(feature_name = feature_name)
    x_tr, y_tr = X[indices_train,:], y[indices_train]
    x_val, y_val = X[indices_val,:], y[indices_val]
    x_te, y_te = X[indices_test,:], y[indices_test]

    for model, hp_list in models.items():
        for hp in hp_list:
            pred_df_col, results_df_row = fit_clf_from_np_modal(x_tr, y_tr, x_val, y_val, x_te, y_te,
                                                                fold_metadata=fold_meta,
                                                  classifier=model,
                                                  classifier_hp=hp,
                                                  input_features_name=feature_name)
            pred_list.append(pred_df_col)
            
            fold_meta_data_df = pl.DataFrame({
                "k_outer": [fold_meta["k_outer"]],
                "k_inner": [fold_meta["k_inner"]],
                "r_outer": [fold_meta["r_outer"]],
                "r_inner": [fold_meta["r_inner"]]
            })
            
            results_df_row = pl.concat([results_df_row, fold_meta_data_df], how="horizontal") 
            results_list.append(results_df_row)
    return pred_list, results_list

def load_arrays_from_mount_local(feature_name:str,mount_directory:str):
        X = np.load(os.path.join(mount_directory,feature_name+".npy"))
        y = np.load(os.path.join(mount_directory,"labels.npy"))
        return X,y
    

def fit_models_local(models: Dict[str, List[Dict[str, Any]]], feature_name: str, indices_train:np.ndarray,indices_val: np.ndarray,indices_test:np.ndarray,
               fold_meta: Dict[str, Any],mount_directory:str) -> Tuple[List[pl.DataFrame], List[pl.DataFrame]]:

    
    pred_list = []
    results_list = []
    X,y = load_arrays_from_mount_local(feature_name = feature_name,mount_directory=mount_directory)
    x_tr, y_tr = X[indices_train,:], y[indices_train]
    x_val, y_val = X[indices_val,:], y[indices_val]
    x_te, y_te = X[indices_test,:], y[indices_test]

    for model, hp_list in models.items():
        for hp in hp_list:
            pred_df_col, results_df_row = fit_clf_from_np_modal(x_tr, y_tr, x_val, y_val, x_te, y_te,
                                                                fold_metadata=fold_meta,
                                                  classifier=model,
                                                  classifier_hp=hp,
                                                  input_features_name=feature_name)
            pred_list.append(pred_df_col)
            
            fold_meta_data_df = pl.DataFrame({
                "k_outer": [fold_meta["k_outer"]],
                "k_inner": [fold_meta["k_inner"]],
                "r_outer": [fold_meta["r_outer"]],
                "r_inner": [fold_meta["r_inner"]]
            })
            
            results_df_row = pl.concat([results_df_row, fold_meta_data_df], how="horizontal") 
            results_list.append(results_df_row)
    return pred_list, results_list

def preprocess_np_modal(df: pl.DataFrame,
                        mount_dir: str ,
                        inputs: List[Dict[str, Union[List[str], List[List[str]]]]],
                        target_column:str = "target") -> Tuple[pl.DataFrame, pl.DataFrame]:
    

    # Preprocess the dataset
    preprocess_start_time = time.time()
    feature_arrays, labels, _ = preprocess_dataset(df, inputs, target_column=target_column)
    #save the arrays to cynde_mount folder
    print(f"Saving arrays to {mount_dir}")
    for feature_name,feature_array in feature_arrays.items():
        np.save(os.path.join(mount_dir,feature_name+".npy"),feature_array)
    np.save(os.path.join(mount_dir,"labels.npy"),labels)
    preprocess_end_time = time.time()
    print(f"Preprocessing completed in {preprocess_end_time - preprocess_start_time} seconds")

def check_preprocessed_np_modal(mount_dir: str, inputs: List[Dict[str, Union[List[str], List[List[str]]]]]) -> bool:
    for feature_dict in inputs:
        for feature_list in feature_dict.values():
            for feature in feature_list:
                if not os.path.exists(os.path.join(mount_dir,feature+".npy")):
                    raise ValueError(f"Feature {feature} not found in {mount_dir}")
    return True

def train_nested_cv_from_np_modal(df: pl.DataFrame,
                        cv_type: Tuple[str, str],
                        mount_dir: str ,
                        inputs: List[Dict[str, Union[List[str], List[List[str]]]]],
                        models: Dict[str, List[Dict[str, Any]]],
                        group_outer: List[str],
                        k_outer: int,
                        group_inner: List[str],
                        k_inner: int,
                        r_outer: int = 1,
                        r_inner: int = 1,
                        run_local: bool = False,
                        target_column:str = "target",
                        load_preprocess:bool=True) -> Tuple[pl.DataFrame, pl.DataFrame]:
    start_time = time.time()
    df = check_add_cv_index(df)
    pred_df = nested_cv(df, cv_type, group_outer, k_outer, group_inner, k_inner, r_outer, r_inner, return_joined=False)
    cv_df = df.join(pred_df, on="cv_index", how="left")
    results_df = pl.DataFrame(schema=RESULTS_SCHEMA)
    print("results schema: ", results_df.schema)
    print(f"results shape: {results_df.shape}")

    # Preprocess the dataset
    preprocess_start_time = time.time()
    if load_preprocess:
        check_preprocessed_np_modal(mount_dir,inputs)
        feature_names = cf.derive_feature_names(inputs)
    else:
        feature_arrays, labels, _ = preprocess_dataset(df, inputs, target_column=target_column)
        #save the arrays to cynde_mount folder
        print(f"Saving arrays to {mount_dir}")
        for feature_name,feature_array in feature_arrays.items():
            np.save(os.path.join(mount_dir,feature_name+".npy"),feature_array)
        np.save(os.path.join(mount_dir,"labels.npy"),labels)
        feature_names = list(feature_arrays.keys())
    preprocess_end_time = time.time()
    print(f"Preprocessing completed in {preprocess_end_time - preprocess_start_time} seconds")

    all_pred_list = []
    all_results_list = []

    # Generate folds and fit models
    folds_generation_start_time = time.time()
    fit_tasks = generate_folds_from_np_modal_compatible(models,cv_df, cv_type, feature_names, group_outer, k_outer, group_inner, k_inner, r_outer, r_inner, mount_dir)
    if not run_local:
        all_tuples_list = fit_models_modal.starmap(fit_tasks)
    else:
        all_tuples_list = []
        for task in fit_tasks:
            all_tuples_list.append(fit_models_local(*task))

    #all_tuples_list is a list of tuples(list(pred_df),list(results_df)) we want to get to a single list
    for (pred_list,res_list) in all_tuples_list:
        all_results_list.extend(res_list)
        all_pred_list.extend(pred_list)
    #flatten the list of lists
    for frame in all_pred_list:
        pred_df = pred_df.join(frame, on="cv_index", how="left")
    folds_generation_end_time = time.time()
    print(f"Folds generation and model fitting completed in {folds_generation_end_time - folds_generation_start_time} seconds")
    print(f"Average time per fold: {(folds_generation_end_time - folds_generation_start_time) / (k_outer * k_inner * r_outer * r_inner)}")



    # Aggregate and save results
    aggregation_start_time = time.time()
    for results_df_row in all_results_list:
        results_df = results_df.vstack(results_df_row)


    aggregation_end_time = time.time()
    print(f"Aggregation of results completed in {aggregation_end_time - aggregation_start_time} seconds")

    save_results_start_time = time.time()
    # cf.save_results(df, results_df, pred_df, save_name, base_path=base_path)
    save_results_end_time = time.time()
    print(f"Saving results completed in {save_results_end_time - save_results_start_time} seconds")

    total_end_time = time.time()
    print(f"Total training and processing time: {total_end_time - start_time} seconds")
    tot_models =0
    for model,hp_list in models.items():
        tot_models += len(hp_list)

    print(f"Total average time per fold: {(total_end_time - start_time) / (k_outer * k_inner * r_outer * r_inner*tot_models*len(inputs))} seconds")

    return results_df, pred_df

---

## embed

import time
from time import perf_counter
import os
from typing import List, Union, Any, Optional
import polars as pl
from cynde.models.embedders import  get_embedding_list
from openai import Client
import json
import tiktoken

import asyncio
from cynde.async_tools.api_request_parallel_processor import process_api_requests_from_file

MAX_INPUT = {
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191
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

# def generate_embedding_batched_payloads_from_column(filename: str, df: pl.DataFrame, column_name: str, model_name: str = "text-embedding-3-small", batch_size:int=100):
#     """Generates embedding payloads from a column in a DataFrame and saves them to a JSONL file."""
#     data = df[column_name].to_list()
#     batch = []
#     with open(filename, "w") as f:
#         for x in data:
#             # Replace single quotes with double quotes and escape inner double quotes and newline characters
#             x = x.replace("'", '"').replace('"', '\\"').replace('\n', '\\n')
#             batch.append(x)
#             # Check if batch size is reached; note the off-by-one adjustment
#             if len(batch) == batch_size:
#                 payload = {"model": model_name, "input": batch}
#                 f.write(json.dumps(payload) + "\n")
#                 batch = []  # Reset batch

#         # Handle last batch if it's not empty
#         if batch:
#             payload = {"model": model_name, "input": batch}
#             f.write(json.dumps(payload) + "\n")

#     # Read the JSONL file content into a list of dictionaries
#     with open(filename, 'r') as f:
#         json_list = [json.loads(line) for line in f]

#     # Convert the list of dictionaries into a DataFrame
#     payloads_df = pl.DataFrame(json_list)

#     # Concatenate the original DataFrame column with the generated payloads DataFrame
#     result_df = pl.concat([df.select(column_name), payloads_df], how="horizontal")

#     return result_df


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

def embed_column(df: pl.DataFrame, column_name: str, requests_filepath: str, results_filepath: str, api_key: str, model_name="text-embedding-3-small", batch_size:int=100) -> pl.DataFrame:
    request_url = "https://api.openai.com/v1/embeddings"
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
            max_requests_per_minute=float(10_000),
            max_tokens_per_minute=float(10_000_000),
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
            
            requests_filepath = os.path.join(os.environ.get('CACHE_DIR'), f"{target_column}_{model_name}_requests.jsonl")
            results_filepath = os.path.join(os.environ.get('OUTPUT_DIR'), f"{target_column}_{model_name}_results.jsonl")
            if os.path.exists(requests_filepath) or os.path.exists(results_filepath):
                time_code = time.strftime("%Y-%m-%d_%H-%M-%S")
                requests_filepath = os.path.join(os.environ.get('CACHE_DIR'), f"{target_column}_{model_name}_{time_code}_requests.jsonl")
                results_filepath = os.path.join(os.environ.get('OUTPUT_DIR'), f"{target_column}_{model_name}_{time_code}_results.jsonl")
                
            
            # Generate embeddings and merge them into the DataFrame
            df = embed_column(
                df=df, 
                column_name=target_column, 
                requests_filepath=requests_filepath, 
                results_filepath=results_filepath, 
                api_key=api_key, 
                model_name=model_name
            )
            
            print(f"Embeddings for column '{target_column}' with model '{model_name}' have been merged into the DataFrame.")
    
    return df

def compute_embedding_price(df, text_column, model):
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

