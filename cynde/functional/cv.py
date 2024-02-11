import polars as pl
import numpy as np
from typing import List, Tuple, Union
import time
from cynde.functional.classification import get_features, fit_clf
import os

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
def train_opis_cv(df:pl.DataFrame,
                  embeddings:List[List[str]],
                  classifier:str,
                  group_outer:List[str],k_outer:int,group_inner:List[str],k_inner:int,r_outer:int =1, r_inner:int =1,
                  save_name:str="results.parquet"):
    out = outer_purged_inner_stratified_cv(df,group_outer,k_outer,group_inner,k_inner,r_outer,r_inner)
    results_df_schema = {"classifier":pl.Utf8,
                         "fold_name":pl.Utf8,
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
                            "r_inner":pl.Int64,
                            "feature_time":pl.Utf8,
                            "total_fold_time":pl.Utf8,}
    results_df = pl.DataFrame(schema = results_df_schema)
    pred_df = out.select(pl.col("cv_index"))
    
    #outer replica loop
    for r_o in range(r_outer):
        for k_o in range(k_outer):
            for r_i in range(r_inner):
                for k_i in range(k_inner):
                    start_fold_time = time.time()
                    fold_name = get_fold_name_ps(group_outer,r_o,k_o,group_inner,r_i,k_i)
                    
                    fold_frame = out.select(pl.col("cv_index"),pl.col(fold_name))
                    df_train = out.filter(pl.col(fold_name)=="train")
                    df_val = out.filter(pl.col(fold_name)=="val")
                    df_test = out.filter(pl.col(fold_name)=="test")
                    for embedding in embeddings:
                        start_feature_time = time.time()
                        x_tr, y_tr = get_features(df_train,embedding,[])
                        x_val, y_val = get_features(df_val,embedding,[])
                        x_te, y_te = get_features(df_test,embedding,[])
                        end_feature_time = time.time()
                        human_readable_feature_time = time.strftime("%H:%M:%S", time.gmtime(end_feature_time-start_feature_time))
                        embedding_name = "_".join(embedding)
                        print(f"Training classifier {classifier} with features {embedding_name}")
                        pred_df_col,results_df_row = fit_clf(x_tr, y_tr, x_val, y_val,x_te, y_te,classifier=classifier,fold_frame=fold_frame,input_features_name=embedding_name)
                        pred_df = pred_df.join(pred_df_col, on="cv_index", how="left")
                        end_fold_time = time.time()
                        human_readable_fold_time = time.strftime("%H:%M:%S", time.gmtime(end_fold_time-start_fold_time))
                        print(f"Total fold time: {human_readable_fold_time}")
                        time_dict = {"feature_time":human_readable_feature_time,"total_fold_time":human_readable_fold_time}
                        fold_meta_data_df = pl.DataFrame({"k_outer":[k_o],"k_inner":[k_i],"r_outer":[r_o],"r_inner":[r_i],"time":[time_dict]}).unnest("time")
                        #hcat the fold_meta_data_df to results_df_row but we unnest time from both first
                        results_df_row = pl.concat([results_df_row,fold_meta_data_df],how="horizontal")                   
                        results_df = results_df.vstack(results_df_row)

    #add_time-stamp to name
    time_stamp_str_url_compatible = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_name_res = "results_"+time_stamp_str_url_compatible+"_"+save_name
    save_name_pred = "predictions_"+time_stamp_str_url_compatible+"_"+save_name
    print(f"Saving results to {save_name_res}")
    print(f"Saving predictions to {save_name_pred}")
    #use os to merge filenames with data_processed folder
    save_name_res = os.path.join("data_processed",save_name_res)
    save_name_pred = os.path.join("data_processed",save_name_pred)
    results_df.write_parquet(save_name_res)
    pred_df.write_parquet(save_name_pred)
    joined_df = df.join(out, on="cv_index", how="left")
    joined_df = joined_df.join(pred_df, on="cv_index", how="left")
    save_name_joined_df = "joined_"+time_stamp_str_url_compatible+"_"+save_name
    save_name_joined_df = os.path.join("data_processed",save_name_joined_df)
    joined_df.write_parquet(save_name_joined_df)
    return results_df,pred_df