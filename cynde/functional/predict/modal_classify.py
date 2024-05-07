import modal
from typing import Tuple
from cynde.functional.predict.types import PipelineResults,PredictConfig
from cynde.functional.predict.preprocess import check_add_cv_index
from cynde.functional.predict.cv import generate_nested_cv
import polars as pl



def train_nested_cv_distributed(df:pl.DataFrame,task_config:PredictConfig) -> pl.DataFrame:
    """ Deploy a CV training pipeline to Modal, it requires a df with cv_index column and the features set to have already pre-processed and cached 
    1) Validate the input_config and check if the preprocessed features are present locally 
    2) create a generator that yields the modal path to the features and targets frames as well as the scikit pipeline object 
    3) execute through a modal starmap a script that fit end eval each pipeline on each feature set and return the results
    4) collect and aggregate the results locally and save and return the results
    """
    #validate the inputs and check if the preprocessed features are present locally
    df = check_add_cv_index(df,strict=True)
    
    f = modal.Function.lookup(task_config.modal_endpoint, "predict_pipeline_distributed")
    
    #extract the subset of columns necessary for constructing the cross validation folds 
    unique_groups = list(set(task_config.cv_config.inner.groups + task_config.cv_config.outer.groups))
    df_idx = df.select(pl.col("cv_index"),pl.col(unique_groups))

    nested_cv = generate_nested_cv(df_idx,task_config)
    all_results = []
    for result in f.map(list(nested_cv)):
        all_results.append(result)
    re_validated_results = []
    for result in all_results:
        re_validated_results.append(PipelineResults.model_validate(result))
    print("Finished!! " ,len(all_results))