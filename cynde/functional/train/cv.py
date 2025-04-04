import polars as pl
from pydantic import BaseModel
from typing import List, Optional, Tuple, Generator
import itertools
from cynde.functional.train.types import FoldMeta,PredictConfig, BaseFoldConfig,PipelineInput,BaseClassifierConfig,ClassifierConfig,InputConfig,CVConfig, KFoldConfig, PurgedConfig, StratifiedConfig, CVSummary

from cynde.functional.train.preprocess import check_add_cv_index




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

def hacky_list_relative_slice(input_list: List[int],k: int):
    slices = {}
    slice_size = len(input_list) // k
    for i in range(k):
        if i < k - 1:
            slices["fold_{}".format(i)] = input_list[i*slice_size:(i+1)*slice_size]
        else:
            # For the last slice, include the remainder
            slices["fold_{}".format(i)] = input_list[i*slice_size:]
    return slices

def get_sliced_frame(sdf: pl.DataFrame,k: int,target_col: str = "cv_index"):
    slices_dicts = []
    for row in sdf.iter_rows(named=True):
        cv_index = row[target_col]
        sliced_cv_index = hacky_list_relative_slice(cv_index,k)
        slices_dicts.append(sliced_cv_index)
    return pl.DataFrame(slices_dicts)

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
        test_series = pl.concat([folds[i] for i in test_fold])
        train_series = pl.concat([folds[i] for i in range(config.k) if i not in test_fold])
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
    gdf = df.group_by(config.groups).agg(pl.col("cv_index")).select(pl.col(config.groups+["cv_index"]))
    gdf_slices = slice_frame(gdf,config.k,shuffle=config.shuffle,explode=True)
    train_indexes = []
    test_indexes = []
    fold_numbers = []
    test_folds = list(itertools.combinations(range(config.k),config.n_test_folds))
    for fold_number, test_fold in enumerate(test_folds):
        test_series = pl.concat([gdf_slices[i] for i in test_fold])["cv_index"]
        train_series = pl.concat([gdf_slices[i] for i in range(config.k) if i not in test_fold])["cv_index"]
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
    gdf = df.group_by(config.groups).agg(pl.col("cv_index")).select(pl.col(config.groups+["cv_index"]))
    train_indexes = []
    test_indexes = []
    montecarlo_replicas = []
    for i in range(config.montecarlo_replicas):
        gdf_slices = slice_frame(gdf,config.k,shuffle=True,explode=True)
        train_series = pl.concat(gdf_slices[:config.k-config.n_test_folds])["cv_index"]
        test_series = pl.concat(gdf_slices[config.k-config.n_test_folds:])["cv_index"]
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
    sliced = get_sliced_frame(sdf,k,target_col="cv_index")
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
                    fold_meta = FoldMeta(r_inner=r_i,k_inner=k_i,r_outer=r_o,k_outer=k_o)
                    df_val_idx_o_i = df_idx.filter(pl.col("cv_index").is_in(val_idx_i))
                    df_train_idx_o_i = df_idx.filter(pl.col("cv_index").is_in(train_idx_i))
                    n_train = df_train_idx_o_i.shape[0]
                    n_val = df_val_idx_o_i.shape[0]
                    n_test = df_test_idx_o.shape[0]
                    print(f"For outer replica {r_o}, outer fold {k_o}, inner replica {r_i}, inner fold {k_i}: train {n_train}, val {n_val}, test {n_test} samples.")
                    #Feature types loop
                    for feature_index,feature_set in enumerate(input_config.feature_sets):
                        for classifier in classifiers_config.classifiers:
                            yield PipelineInput(train_idx=df_train_idx_o_i,val_idx= df_val_idx_o_i,test_idx= df_test_idx_o,feature_index= feature_index,cls_config= classifier,input_config=input_config,fold_meta=fold_meta)