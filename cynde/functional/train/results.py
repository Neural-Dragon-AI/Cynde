from typing import List
from cynde.functional.train.types import PipelineResults
import polars as pl

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