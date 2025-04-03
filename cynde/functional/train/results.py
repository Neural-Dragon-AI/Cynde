from typing import List, Optional, Tuple, Generator
from cynde.functional.train.types import PipelineResults
import polars as pl
import time
import os

def merge_results(results:List[PipelineResults]) -> pl.DataFrame:
    """Merge the results of the nested cv into a single dataframe joining over the cv_index column"""
    dfs = [result.to_results_df() for result in results]
    merged_df = dfs[0]
    for to_join in dfs[1:]:
        merged_df = merged_df.join(to_join,on="cv_index",how="left")
    return merged_df

def merge_metrics(results:List[PipelineResults]) -> pl.DataFrame:
    """Merge the results of the nested cv into a single dataframe joining over the cv_index column"""
    dfs = [result.to_metrics_df() for result in results]
    metric_df = pl.concat(dfs,how="vertical")
    return metric_df

def aggr_metrics_by_inputs(metrics_df:pl.DataFrame):
    metrics_columns = ["train_accuracy","train_mcc","val_accuracy","val_mcc","test_accuracy","test_mcc"]
    return metrics_df.group_by("feature_set_name","classifier_id").agg(pl.col(metrics_columns).mean(),pl.col(metrics_columns).std().name.suffix("_std"),pl.len())

def save_results(results:List[PipelineResults],name_prefix:Optional[str] = None):
    #extract from first input the save_path
    save_path = results[0].pipeline_input.input_config.save_folder
    results_df = merge_results(results)
    metrics_df = merge_metrics(results)
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    save_name_results = f"results_{time_stamp}.parquet" if name_prefix is None else f"{name_prefix}_results_{time_stamp}.parquet"
    save_name_metrics = f"metrics_{time_stamp}.parquet" if name_prefix is None else f"{name_prefix}_metrics_{time_stamp}.parquet"
    save_path_results = os.path.join(save_path,save_name_results)
    save_path_metrics = os.path.join(save_path,save_name_metrics)
    print(f"Results saved at {save_path_results}")
    print(f"Metrics saved at {save_path_metrics}")
    results_df.write_parquet(save_path_results)
    metrics_df.write_parquet(save_path_metrics)

    
  