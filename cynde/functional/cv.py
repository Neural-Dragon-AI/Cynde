import polars as pl
import numpy as np
from typing import List, Tuple, Union


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


def purged_kfold(df:pl.DataFrame,group:List[str],k:int,shuffle:bool=True,pre_name:str="",target_names:Tuple[str,str] = ("train","test")):
    #group is a list of columns that will be used to group the data
    #k is the number of splits
    #we will use the group to split the data into k groups and then we will use the group to make sure that the same group is not in the train and test set
    #we will use the row index to split the data
    #first we will shuffle the data
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

def outer_purged_inner_stratified_cv(df:pl.DataFrame,group_outer:List[str],k_outer:int,group_inner:List[str],k_inner:int,r_outer:int =1, r_inner:int =1):
     #create a name for group_outer and group_inner concatenating the names
    #check if df has a cv_index column if not create one
    if "cv_index" not in df.columns:
        df = df.with_row_index(name="cv_index")
    name_group_outer = "_".join(group_outer)
    name_group_inner = "_".join(group_inner)
    selected_columns = ["cv_index"]+group_outer+group_inner
    
    df_joined = df.select(pl.col(selected_columns))
    for r_out in range(r_outer):
        outer_pre_name = "outer_purged_{}_replica_{}_".format(name_group_outer,r_out)
        
        purged_folds_df = purged_kfold(df,group_outer,k_outer,shuffle=True,pre_name=outer_pre_name,target_names=("dev","test"))
        
        df_joined = df_joined.join(purged_folds_df, on="cv_index", how="left")

            #join back purged_folg_df to gdf
        # new_joined = df_joined
        for outer_index in range(k_outer):
            for r_in in range(r_inner):
                inner_pre_name = "inner_stratified_{}_replica_{}_".format(name_group_inner,r_in)
                stratifed_folds = stratified_kfold(df_joined.filter(pl.col("{}fold_{}".format(outer_pre_name,outer_index))=="dev"),group_inner,k_inner,shuffle=True,pre_name="{}fold_{}_{}".format(outer_pre_name,outer_index,inner_pre_name),target_names=("train","val"))
                df_joined = df_joined.join(stratifed_folds, on="cv_index", how="left").fill_null("test")
    return df.join(df_joined, on="cv_index", how="left")
    
def get_fold_name_ps(group_purged:List[str],
                        replica_purged:int,
                        fold_purged:int,
                        group_stratified:List[str],
                        replica_stratified:int,
                        fold_stratified:int):
    return "outer_purged_{}_replica_{}_fold_{}_inner_stratified_{}_replica_{}_fold_{}".format("_".join(group_purged),
                                                                                              replica_purged,
                                                                                              fold_purged,
                                                                                              "_".join(group_stratified),
                                                                                              replica_stratified,
                                                                                              fold_stratified)
