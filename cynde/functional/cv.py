import polars as pl
import numpy as np
from typing import List, Tuple, Union, Dict, Any
import time
from cynde.functional.classify import get_features, fit_clf
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
def get_cv_function(cv_type:str):
    print(f"cv_type: {cv_type}")
    if cv_type == "purged":
        return purged_kfold
    elif cv_type == "stratified":
        return stratified_kfold
    elif cv_type == "vanilla":
        return vanilla_kfold
    else:
        raise ValueError("cv_type can only be purged, stratified or vanilla")

def validate_cv_type(cv_type: Tuple[str,str]):
    for cv_subtype in cv_type:
        if cv_subtype not in ["purged","stratified","vanilla"]:
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
                complete_pre_name = "{}fold_{}_{}".format(outer_pre_name,k_out,inner_pre_name)
                inner_folds = inner_cv_function(dev_df,group_inner,k_inner,shuffle=True,pre_name=complete_pre_name,target_names=("train","val"))
                cv_df = cv_df.join(inner_folds, on="cv_index", how="left").fill_null("test")
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
                    save_name:str="nested_cv_out") -> Tuple[pl.DataFrame,pl.DataFrame]:
    
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
                                                                     classifer_hp= hp,
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
    up_one = os.path.join("..","data_processed")
    save_name_res = os.path.join(up_one,save_name_res)
    save_name_pred = os.path.join(up_one,save_name_pred)
    save_name_joined_df = os.path.join(up_one,save_name_joined_df)
    print(f"Saving results to {save_name_res}")
    results_df.write_parquet(save_name_res)
    pred_df.write_parquet(save_name_pred)
    df.join(pred_df, on="cv_index", how="left").write_parquet(save_name_joined_df)
    return results_df,pred_df



