# 🌿✨ Cynde: A Framework for Scalable and Flexible LLM-Powered Data Processing

## Introduction

Cynde is a Python framework designed to streamline the integration of large language models (LLMs) with the modern data science stack, such as Polars DataFrames and decision tree ensembles, for efficient and scalable data processing. The framework provides a set of modular and composable tools for tasks such as embedding generation, structured text generation, and predictive modeling, with a focus on leveraging the power of LLMs and serverless computing.

Cynde offers two main behavioral patterns for interacting with LLMs:

1. Compatibility with OpenAI API-compatible servers, including popular LLMs like OpenAI's GPT models and self-hosted solutions like llama.cpp or OpenLLaMA
2. Self-hosted deployments using open-source models like Meta's LLaMA and Hugging Face's text-generation-inference (TGI) and text-embedding-inference (TEI) servers, managed by Cynde through Modal

The framework is designed to be flexible and adaptable to various LLM backends, allowing users to choose the most suitable option for their needs, whether it's a cloud-based API or a self-hosted solution.

Cynde integrates with Logfire, an observability platform, to provide powerful monitoring and insights into the performance and behavior of LLM-powered data processing pipelines. By leveraging Logfire's automatic instrumentation and Pydantic model recording capabilities, Cynde enables users to gain deep visibility into the execution and performance of their workflows, making it easier to identify bottlenecks, optimize resource utilization, and ensure the reliability of their applications.

One of the key features of Cynde is its ability to match Polars' parallel nature, either by scaling to the maximum tokens/calls limit of the OpenAI-compatible server (hence the limit becomes the rate limit of the provider) or by exploiting the highly optimized TGI and TEI servers, which use Rust-based LLM inference with continuous batching.

TGI and TEI are toolkits for deploying and serving LLMs, enabling high-performance text generation and embedding for the most popular open-source models. They implement various optimizations and features, such as tensor parallelism for faster inference on multiple GPUs, token streaming using Server-Sent Events (SSE), continuous batching of incoming requests for increased total throughput, and optimized transformers code for inference using Flash Attention and Paged Attention on the most popular architectures.

By leveraging Modal's autoscaling capabilities, Cynde can efficiently scale the TGI and TEI deployments to handle large workloads, with the scaling limits being determined by the available resources on the Modal cloud platform.

## Key Features

### OpenAI API-Compatible Server Integration

Cynde is compatible with any OpenAI API-compatible server, including popular LLMs like OpenAI's GPT models and self-hosted solutions like llama.cpp or OpenLLaMA. This allows users to leverage the power of LLMs while maintaining flexibility in their deployment options.

### Serverless LLM Integration

Cynde seamlessly integrates with serverless computing platforms like Modal for massively parallel LLM processing. It supports self-hosted deployments using open-source models like Meta's LLaMA and Hugging Face's TGI and TEI servers. These self-hosted deployments are managed by Cynde through Modal, ensuring scalability and efficiency.

### Polars DataFrames

Built on top of Polars, a fast and efficient DataFrame library for Rust and Python, Cynde leverages lazy evaluation and query optimization to efficiently process large-scale data. Cynde's functional API enables easy composition of DataFrame operations, allowing for expressive and efficient data transformations.

### Pydantic Models

Cynde uses Pydantic, a Python library for data parsing and validation using Python type annotations, to define and validate data structures throughout the framework. Pydantic models are used for configuring LLM interactions, specifying data schemas, and defining pipeline steps, ensuring type safety and reducing the potential for errors.

### Functional API

Cynde exposes a functional API where methods take a DataFrame and a Pydantic object as input and return a DataFrame, a Pydantic object, or a tuple of both. This allows for easy composition of functions and enables thinking in terms of type-based morphisms, making the code more readable, maintainable, and less prone to errors.

The functional API revolves around three main modules:

1. `cynde.functional.embed`: Provides tools for generating embeddings from text data using LLMs
2. `cynde.functional.generate`: Offers functionality for generating structured text using LLMs based on specified instructions and output schemas
3. `cynde.functional.predict`: Enables predictive modeling tasks, focusing on categorical variables, with support for distributed nested cross-validation of tree ensembles

These modules are designed to work seamlessly with Polars DataFrames, allowing for efficient and expressive data processing pipelines.

```mermaid
graph LR
    A[DataFrame<br>str] --> B[cynde.functional.embed]
    A --> C[cynde.functional.generate]
    E[DataFrame<br>enum] --> C
    T[DataFrame<br>struct] --> C
    E --> F[cynde.functional.predict.train]
    D[DataFrame<br>float] --> F
    G[DataFrame<br>list_float] --> F

    B --> N[DataFrame<br>list_float]

    C --> T2[DataFrame<br>struct]
    C --> A2[DataFrame<br>str]
    C --> E2[DataFrame<br>enum]

    F --> X[cynde.functional.predict.predict]
    X --> E2
```

The diagram above illustrates the high-level design of Cynde's functional API, showcasing the input and output types of each module and the flow of data between them. By leveraging the strong typing of Polars DataFrames and Pydantic models, Cynde enables expressive and type-safe data processing pipelines that are easy to reason about and maintain.

One of the key features of Cynde's functional API is its support for typed column transformations. By using Pydantic models to define the expected input and output types of each function, Cynde can perform type-safe transformations on DataFrame columns, ensuring that the data flowing through the pipeline conforms to the expected schema.

For example, the `cynde.functional.embed` module takes a DataFrame with a string column as input and returns a DataFrame with a list of floats column, representing the embeddings generated by the LLM. Similarly, the `cynde.functional.generate` module takes a DataFrame with string, enum, or struct columns as input and returns a DataFrame with the corresponding generated text, which can be of type string, enum, or struct.

This type-safe approach to column transformations helps prevent errors and inconsistencies in the data pipeline, making it easier to reason about the flow of data and the expected outputs of each step.

## Serverless Deployment and Autoscaling

Cynde leverages Modal for serverless deployment and autoscaling of LLM processing. The framework provides deployment scripts for spinning up TGI and TEI servers on Modal, allowing for efficient and scalable processing of text generation and embedding tasks.

The general design pattern is to push local data to the cloud, where the embedding and text generation servers are deployed. The framework maps the generation and embedding jobs over rows of the DataFrame, with autoscaling to handle large workloads.

```mermaid
graph LR
    A[DataFrame<br>str] --> B[cynde.functional.embed]
    A --> C[cynde.functional.generate]
    E[DataFrame<br>enum] --> C
    T[DataFrame<br>struct] --> C
    E --> F[cynde.functional.predict.train]
    D[DataFrame<br>float] --> F
    G[DataFrame<br>list_float] --> F

    H[Pydantic Model] --> C

    B --> I[JSON Caching]
    B --> J[Modal Deploy TEI]
    I --> K[OpenAI Compatible API]
    J --> L[Remote Inference TEI]
    K --> M[JSON Caching]
    L --> N[DataFrame<br>list_float]
    M --> N

    C --> O[JSON Caching]
    C --> P[Modal Deploy TGI]
    O --> Q[OpenAI Compatible API]
    P --> R[Remote Inference TGI]
    Q --> S[JSON Caching]
    R --> T2[DataFrame<br>struct]
    R --> A2[DataFrame<br>str]
    R --> E2[DataFrame<br>enum]
    S --> T2
    S --> A2
    S --> E2

    F --> U[Modal Deploy TrainSK]
    U --> V[Save in Modal Volume]
    V --> W[Modal Deploy PredictSK]
    F --> X[cynde.functional.predict.predict]
    W --> Z[Remote Inference PredictSK]
    X --> W
    Z --> E2
```

The diagram above illustrates the serverless deployment and autoscaling architecture of Cynde. The framework deploys TGI and TEI servers on Modal, which can be accessed via remote inference endpoints. The `cynde.functional` modules interact with these endpoints to perform embedding generation, structured text generation, and predictive modeling tasks.

For example, to deploy a TGI server on Modal, you can use the provided deployment script:

```bash
modal deploy tgi.py
```

Once the server is deployed, you can call the `generate` function from your local Python environment using Pydantic models:

```python
from cynde.functional.generate.modal_gen import generate_column, validate_df
from cynde.functional.generate.types import InstructionConfig

instruction = InstructionConfig(
    system_prompt="Generate a JSON object describing the following text:",
    column="text",
    output_schema=OutputSchema.model_json_schema(),
    modal_endpoint="example-tgi-endpoint"
)

out_df = generate_column(df, instruction)
validated_df = validate_df(out_df, OutputSchema)
```

This example demonstrates how to use the `generate_column` function to generate structured text from a DataFrame column using a deployed TGI server on Modal. The `InstructionConfig` Pydantic model is used to specify the system prompt, input column, output schema, and the Modal endpoint to use. The resulting DataFrame is then validated against the specified output schema using the `validate_df` function.

Similarly, for embedding generation, you can use the `embed_column` function with a deployed TEI server on Modal:

```python
from cynde.functional.embed.modal_embed import embed_column, EmbedConfig

embed_cfg = EmbedConfig(
    column="text",
    modal_endpoint="example-tei-endpoint"
)

embedded_df = embed_column(df, embed_cfg)
```

This example shows how to generate embeddings for a DataFrame column using a deployed TEI server on Modal. The `EmbedConfig` Pydantic model is used to specify the input column and the Modal endpoint to use.

By leveraging Modal's serverless deployment and autoscaling capabilities, Cynde can efficiently process large-scale data and handle varying workloads without the need for manual infrastructure management.

## Refactoring Steps

To further enhance the framework and provide a more consistent and flexible interface for users, the following refactoring steps are planned:

### OpenAI API-Compatible Server Integration

Refactor the `cynde.functional.embed` and `cynde.functional.generate` modules to support OpenAI API-compatible servers, including self-hosted solutions like llama.cpp or OpenLLaMA. This will allow users to leverage the power of LLMs while maintaining flexibility in their deployment options.

Example usage for embedding generation with an OpenAI API-compatible server:

```python
from cynde.functional.embed import embed_column
from cynde.functional.embed.types import EmbedConfig

embed_cfg = EmbedConfig(
    column="text",
    api_endpoint="https://api.openai.com/v1/embeddings",
    api_key="your_api_key",
    model="text-embedding-ada-002"
)

embedded_df = embed_column(df, embed_cfg)
```

Example usage for structured text generation with an OpenAI API-compatible server:

```python
from cynde.functional.generate import generate_column, validate_df
from cynde.functional.generate.types import InstructionConfig

instruction = InstructionConfig(
    system_prompt="Generate a JSON object describing the following text:",
    column="text",
    output_schema=OutputSchema.model_json_schema(),
    api_endpoint="https://api.openai.com/v1/completions",
    api_key="your_api_key",
    model="text-davinci-003"
)

out_df = generate_column(df, instruction)
validated_df = validate_df(out_df, OutputSchema)
```

### Unify Generation and Embedding API

Refactor the `cynde.functional.generate` and `cynde.functional.embed` modules to use a higher-level Pydantic config that specifies the backend to use (e.g., OpenAI API, Modal deployment). This will provide a more consistent and flexible interface for users.

### Refactor Predict Module

Refactor the `cynde.functional.predict` module to work through the Modal deployment invocation instead of the current `main()` function. Use Modal volumes for handling training data instead of the current mount system, allowing for more efficient data handling and reduced networking overhead.

Example usage for predictive modeling with the refactored `cynde.functional.predict` module:

```python
from cynde.functional.predict import train_predict_pipeline
from cynde.functional.predict.types import PredictConfig

predict_cfg = PredictConfig(
    input_config=input_config,
    cv_config=cv_config,
    classifiers_config=classifiers_config,
    modal_endpoint="example-predict-endpoint"
)

results_df = train_predict_pipeline(df, predict_cfg)
```

In this example, the `train_predict_pipeline` function is used to perform predictive modeling on a DataFrame using a deployed prediction endpoint on Modal. The `PredictConfig` Pydantic model is used to specify the input configuration, cross-validation configuration, classifiers configuration, and the Modal endpoint to use.

By using Modal volumes for handling training data, the refactored predict module can efficiently process large datasets and perform distributed cross-validation without the need for data serialization and transfer for each row. This approach optimizes data handling and reduces networking overhead, especially in scenarios where the same rows are used across multiple folds.

### Add Storage and Inference Endpoint

Extend the Modal deployment to include a storage and inference endpoint for the gradient boosting trees trained in the `train_cv` endpoint. This will enable seamless model persistence and serving, allowing for efficient model storage and retrieval during inference.

## Pipeline with Remote References

The next step in the evolution of Cynde is to introduce a pipeline mechanism that allows for the composition of methods through eager invocation from the local machine, where the original data resides. Each step in the pipeline would return a reference to the data that will be used as input for the next step.

To achieve this, we propose introducing a dummy class that represents a reference to a (potentially) remote DataFrame that does not exist in advance. This class would encapsulate the metadata required to locate and load the actual data when needed.

Here's how this pipeline mechanism would work:

1. Each step in the pipeline is triggered from the local machine, where the original data resides.
2. When a step is executed, it returns a dummy object that represents the output DataFrame, rather than the actual data itself.
3. This dummy object contains metadata about the expected remote location and name of the output DataFrame.
4. When the dummy object is passed as input to the next step in the pipeline, the Modal execution environment uses the metadata to locate and load the actual remote DataFrame.

By leveraging this remote reference mechanism, Cynde can enable the composition of complex data processing pipelines that span multiple execution environments (e.g., local machine, Modal) without the need to transfer data back and forth between steps. This approach also allows for lazy evaluation and optimization of the pipeline, as the actual data is only loaded and processed when needed.

To implement this pipeline mechanism, we will need to define a `RemoteDataFrame` class that encapsulates the metadata required to locate and load the remote data. This class will provide methods for saving and loading data to and from remote storage, as well as for specifying the expected schema and transformations applied to the data.

By integrating this remote reference mechanism with the refactored `cynde.functional` modules and the Modal deployment infrastructure, Cynde will provide a powerful and flexible framework for composing and executing complex data processing pipelines that leverage the power of LLMs and serverless computing.
