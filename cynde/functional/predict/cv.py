import polars as pl
from typing import List, Optional
import itertools
from cynde.functional.predict.types import CVConfig, KFoldConfig, PurgedConfig, StratifiedConfig, CVSummary

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