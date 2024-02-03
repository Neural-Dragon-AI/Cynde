# Cynde
### Integrating Semantic Wisdom with Predictive Models

python -m pip install pathto/Cynde

Cynde, inspired by the Old English word  /ˈkyn.de/ for something inherent or innate, is crafted to tackle the complex challenge of integrating unstructured text with structured tabular data. By utilizing the collective intelligence of large language models (LLMs), Cynde refines the approach to enrich, analyze, and model data within Polars data frames. LLMs outputs are tamed via grammar-constrained extraction to adhere to pydantic data-classes that encapsulate the user domain specific knowledge and objectives. Cynde allows further combines the generated data with traditional tabular data for predictive modelling or can use them as target variables for distillation of the llm capabilities into smaller models using a combination of language embeddings and decision trees ensembles.
This library stands at the crossroads of the large language model craze and contemporary data science, enabling a sophisticated fusion of alchemic LLMs procedures with cutting-edge predictive modeling techniques. 

## Core Functionalities

### Semantic Embeddings from Textual Data
- Cynde allows for the automatic creation of new array columns in Polars data frames, converting string columns into semantic embeddings. This is achieved through the use of encoder-only transformers models mapping textual information into a numerical format that is immediately useful for machine learning applications.

    ```python
    import polars as pl
    import cynde.functional as cf 
    from datetime import datetime

    # Sample data frame initialization
    df = pl.DataFrame(
        {
            "customer_id": [101, 102, 103],
            "feedback": [
                "Loved the new product line!",
                "The service was disappointing this time.",
                "Great experience with customer support.",
            ],
            "ratings": [[4, 5, 5], [2, 3, 2], [5, 4, 5]],
            "timestamp": [
                datetime(2023, 1, 1, 14, 30),
                datetime(2023, 1, 2, 9, 15),
                datetime(2023, 1, 3, 18, 45),
            ],
        }
    )
    print(df)
    ┌─────────────┬───────────────────────────────────┬───────────┬─────────────────────┐
    │ customer_id ┆ feedback                          ┆ ratings   ┆ timestamp           │
    │ ---         ┆ ---                               ┆ ---       ┆ ---                 │
    │ i64         ┆ str                               ┆ list[i64] ┆ datetime[μs]        │
    ╞═════════════╪═══════════════════════════════════╪═══════════╪═════════════════════╡
    │ 101         ┆ Loved the new product line!       ┆ [4, 5, 5] ┆ 2023-01-01 14:30:00 │
    │ 102         ┆ The service was disappointing th… ┆ [2, 3, 2] ┆ 2023-01-02 09:15:00 │
    │ 103         ┆ Great experience with customer s… ┆ [5, 4, 5] ┆ 2023-01-03 18:45:00 │
    └─────────────┴───────────────────────────────────┴───────────┴─────────────────────┘

    embedded_df = cf.embed_columns(df, ["feedback"], client=client)
    print(embedded_df)

    Creating embeddings for column feedback
    Processing 3 chunks of text in a single batch
    Embedding Processing took 0.5198197364807129 seconds
    shape: (3, 5)
    ┌─────────────┬─────────────────────────┬───────────┬─────────────────────┬────────────────────────┐
    │ customer_id ┆ feedback                ┆ ratings   ┆ timestamp           ┆ feedback_text-embeddin │
    │ ---         ┆ ---                     ┆ ---       ┆ ---                 ┆ g-3-small_…            │
    │ i64         ┆ str                     ┆ list[i64] ┆ datetime[μs]        ┆ ---                    │
    │             ┆                         ┆           ┆                     ┆ list[f64]              │
    ╞═════════════╪═════════════════════════╪═══════════╪═════════════════════╪════════════════════════╡
    │ 101         ┆ Loved the new product   ┆ [4, 5, 5] ┆ 2023-01-01 14:30:00 ┆ [0.029205, -0.036287,  │
    │             ┆ line!                   ┆           ┆                     ┆ … 0.000765…            │
    │ 102         ┆ The service was         ┆ [2, 3, 2] ┆ 2023-01-02 09:15:00 ┆ [-0.005782, 0.019236,  │
    │             ┆ disappointing th…       ┆           ┆                     ┆ … -0.00427…            │
    │ 103         ┆ Great experience with   ┆ [5, 4, 5] ┆ 2023-01-03 18:45:00 ┆ [-0.014194, -0.027349, │
    │             ┆ customer s…             ┆           ┆                     ┆ … 0.02145…             │
    └─────────────┴─────────────────────────┴───────────┴─────────────────────┴────────────────────────┘
    ```
    

### Dynamic Prompt Construction with Polars Expressions and Operations
-  Cynde's advanced use of `polars.format` goes beyond simple text integration, enabling the execution of sophisticated operations on data frame columns to enrich LLM prompts with processed data insights. By incorporating Polars expressions that perform computations or extract specific data attributes, Cynde facilitates the creation of highly informative and contextually rich prompts for LLM interaction.

    This functionality allows for dynamic prompt generation that not only integrates raw data but also leverages the transformed outputs of various column operations—such as aggregations, computations, and temporal data manipulations. Executing these operations in parallel across the dataset, Cynde ensures efficient generation of enriched prompts, perfectly suited for detailed LLM analysis or data augmentation.
    ```python
    import polars as pl


    fstring = "Customer ID: {} provided feedback at {} with ratings {} an average rating of {} with a global mean of {}: '{}'"
    # Dynamic prompt generation with in-select computations

    df_prompted = cf.prompt(embedded_df, 
                        fstring,
                        [pl.col("customer_id"),
                        pl.col("timestamp").dt.hour(), #from timestamp to hour
                        pl.col("ratings").list.eval(pl.element().cast(pl.Utf8)).list.join("-"), #needs to convert list columns to string
                        pl.col("ratings").list.mean(), #from list to float
                        pl.col("ratings").list.mean().mean(), #constant that gets broadcasted with pl.lit
                        pl.col("feedback")],
                        "customer_prompt")
    print(df_prompted)
    shape: (3, 6)
    ┌─────────────┬───────────────────┬───────────┬──────────────┬──────────────────┬──────────────────┐
    │ customer_id ┆ feedback          ┆ ratings   ┆ timestamp    ┆ feedback_text-em ┆ customer_prompt  │
    │ ---         ┆ ---               ┆ ---       ┆ ---          ┆ bedding-3-small_ ┆ ---              │
    │ i64         ┆ str               ┆ list[i64] ┆ datetime[μs] ┆ …                ┆ str              │
    │             ┆                   ┆           ┆              ┆ ---              ┆                  │
    │             ┆                   ┆           ┆              ┆ list[f64]        ┆                  │
    ╞═════════════╪═══════════════════╪═══════════╪══════════════╪══════════════════╪══════════════════╡
    │ 101         ┆ Loved the new     ┆ [4, 5, 5] ┆ 2023-01-01   ┆ [0.029205,       ┆ Customer ID: 101 │
    │             ┆ product line!     ┆           ┆ 14:30:00     ┆ -0.036287, …     ┆ provided feedba… │
    │             ┆                   ┆           ┆              ┆ 0.000765…        ┆                  │
    │ 102         ┆ The service was   ┆ [2, 3, 2] ┆ 2023-01-02   ┆ [-0.005782,      ┆ Customer ID: 102 │
    │             ┆ disappointing th… ┆           ┆ 09:15:00     ┆ 0.019236, …      ┆ provided feedba… │
    │             ┆                   ┆           ┆              ┆ -0.00427…        ┆                  │
    │ 103         ┆ Great experience  ┆ [5, 4, 5] ┆ 2023-01-03   ┆ [-0.014194,      ┆ Customer ID: 103 │
    │             ┆ with customer s…  ┆           ┆ 18:45:00     ┆ -0.027349, …     ┆ provided feedba… │
    │             ┆                   ┆           ┆              ┆ 0.02145…         ┆                  │
    └─────────────┴───────────────────┴───────────┴──────────────┴──────────────────┴──────────────────┘
    ```

### Enhanced Data Columns via Language Model Generation with Cynde

Cynde offers a sophisticated mechanism for enriching Polars data frames with data columns generated by interactions with language model servers. This process is compatible with servers that adhere to the OpenAI API standards, enabling seamless integration with both cloud-provided and locally hosted language model servers. The enhancements brought about by this integration can take multiple forms:

- **Textual Outputs:** Cynde can invoke language models in chat completion mode to append direct natural language responses to data frames, facilitating the enrichment of datasets with nuanced textual information.
  
- **Structured Data:** More than just appending textual data, Cynde enables the transformation of language model outputs into structured data. This is achieved through the use of Pydantic models, which allow users to define their own abstractions and ensure that the model's output conforms to a structured JSON format adhering to these predefined schemas. This functionality is particularly useful for scientific tasks e.g. analyzing compound names for toxic risk assessment or categorizing psychological characteristics in developmental studies.

One of the key features of Cynde is its ability to automate data labeling through AI-generated categorical columns. This process leverages the inherent categorization capabilities of AI to reduce manual annotation efforts significantly and intuitively enhance the structure of the data.

To ensure efficiency and reliability in processing, Cynde employs asynchronous threads for language model interactions. These threads facilitate the offloading of computationally expensive tasks to disk, thereby not obstructing further data frame processing. This asynchronous approach also includes robust error handling mechanisms, automatically managing retries for both extraction validation and API failures, as well as dealing with token-based rate limits. The integration of these features ensures that Cynde can efficiently handle language model processing without impeding the overall data analysis workflow.

This enhancement of data columns through language model generation exemplifies Cynde's commitment to leveraging advanced AI capabilities for data enrichment. By providing tools for seamless integration with language model servers, defining custom data structures through Pydantic models, and ensuring efficient and reliable processing with asynchronous operations, Cynde significantly advances the possibilities for data analysis and enrichment in the realm of predictive modeling.


### Predictive Modeling Integrating Numeric, Tabular and Semantic knowledge.
-  Supporting the development of classifiers that integrate a wide range of traditional data features integrated with semantic knowledge from unstructured columns using: 
    - **High Dimensional Text Embeddings**
    - **Interpretable Categorical Inputs extracted by llms via pydantic abstractions.**
- Distillation of expensive LLM classifiers into to text-embeddings + tree-ensembles.


### Rigorous Evaluation and Feature Tainting via Stratified Nested Cross-Validation

In data science, the integration of unstructured text with structured data, especially when employing pre-trained models and extensive preprocessing pipelines, introduces a myriad of complexities and potential biases. These models, while powerful, often carry the risk of inadvertently introducing data leakage, particularly when features are engineered using techniques like few-shot learning or when utilizing embeddings that capture multi-scale contextual information. For example, generating features with a large language model (LLM) based on few-shot learning can correlate samples in a way that undermines the integrity of separate training and testing folds. Similarly, classifying data at a granular level, based on embeddings that encapsulate broader context, can also lead to subtle forms of bias and leakage. Such scenarios compromise the robustness and generalizability of predictive models by violating the principle that training and evaluation datasets must remain strictly independent.

- **Stratified Nested Cross-Validation** To mitigate these challenges, Cynde introduces a sophisticated tainting system alongside the utilization of stratified nested cross-validation for model evaluation. This approach ensures an unbiased assessment of various prompts and embeddings, crucial for maintaining the integrity of the data analysis process. By employing stratified nested cross-validation, Cynde guarantees that the distribution of data across folds is representative, thus enhancing the reliability and applicability of models developed within this framework.

- **Feature Tainting** The tainting system plays a pivotal role in preserving the sanctity of data separation. It meticulously tracks the causal relationships between inputs and generated features, ensuring that any attempt to predict on data that has causally influenced model weights is promptly identified and halted with a value error. This safeguards against the risk of information leakage, where data from the training set improperly informs the evaluation process. By implementing these rigorous controls, Cynde not only addresses the inherent pitfalls associated with combining extensive data preprocessing and the use of pre-trained models but also establishes a robust framework for developing predictive models that are both accurate and trustworthy.



### Efficiency through Adaptive Learning: A Progressive Distillation Approach

Cynde introduces a novel procedure that enhances model training efficiency by bootstrapping a language model labeling task, which is then progressively distilled into a more efficient model combining embeddings with tree ensemble classifiers, such as random forests. This innovative process strategically reduces the frequency of direct, costly interactions with large language models (LLMs), focusing instead on a two-phase optimization:

1. **Initial Phase:** The process begins with utilizing a LLM to label data, leveraging its comprehensive understanding and analytical capabilities to generate high-quality, contextually rich labels for training data.
   
2. **Progressive Distillation:** As training progresses, Cynde systematically distills the insights gained from the LLM into an embedding plus tree ensemble model. This step is iteratively refined until a satisfactory level of accuracy is achieved. This distilled model becomes the primary mechanism for generating predictions, significantly reducing the computational cost and enhancing the overall efficiency of the model.

To ensure that the distilled model remains accurate and reliable over time, Cynde employs several innovative strategies:

- **Selective LLM Invocation:** The system is designed to sparingly call upon the LLM for evaluating the distilled model's performance, particularly focusing on instances where the tree ensemble model is predicted to falter. This selective invocation strategy ensures that the LLM's analytical power is harnessed in the most impactful way, maintaining high model accuracy while minimizing computational expense.
  
- **Heuristic Lower Bounds and Calibration:** Cynde explores the development of task-specific heuristics that can serve as a lower bound for error rates, providing a mechanism to track and respond to the model's performance over time. This approach could offer a cost-effective way to maintain model accuracy without the need for constant LLM re-evaluation.
  
- **Out-of-Sample Thresholding:** By measuring the distance of new inputs in the text-embedding space, Cynde can calibrate a threshold to identify inputs that are likely to be out-of-sample for the distilled classifier. When inputs are deemed beyond this threshold, the system can fallback to the LLM, ensuring that the model continues to perform well even as it encounters new or significantly different data.

This adaptive learning approach, with its focus on efficiency and strategic use of LLMs, positions Cynde at the forefront of predictive modeling techniques. It exemplifies a commitment to leveraging the depth of LLM insights while maintaining a pragmatic approach to computational resource management, ensuring that advanced data analysis remains accessible and sustainable.


## In Conclusion

Cynde represents a novel convergence of linguistic heritage and predictive data analytics, facilitating the integration of text data into structured analysis with unprecedented depth and efficiency. Through the Polars ecosystem, it enhances the analytical capabilities of data frames to include complex, semantically rich datasets, addressing traditional inefficiencies in data processing. Cynde paves the way for more nuanced, impactful data analysis and predictive modeling, grounded in the profound knowledge and intuition of language.
