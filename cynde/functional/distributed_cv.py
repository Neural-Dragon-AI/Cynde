from modal import Image
import modal
import polars as pl
import os
import numpy as np
import polars as pl
import time
import os
from typing import Tuple, Optional, Dict, List, Any, Union
from cynde.functional.cv import generate_folds_from_np_modal_compatible, check_add_cv_index, preprocess_dataset, nested_cv, RESULTS_SCHEMA
from cynde.functional.classify import fit_clf_from_np_modal, load_arrays_from_mount_modal

cv_stub = modal.Stub("distributed_cv")
LOCAL_MOUNT_PATH = os.getenv('MODAL_MOUNT')
    
datascience_image = (
    Image.debian_slim(python_version="3.12.1")
    .apt_install("git")
    .pip_install("polars","scikit-learn","openai","tiktoken", force_build=True)
    
    .run_commands("git clone https://github.com/Neural-Dragon-AI/Cynde/")
    .env({"CYNDE_DIR": "/opt/cynde"})
    .run_commands("cd Cynde && pip install -r requirements.txt && pip install .")
)
with datascience_image.imports():
    import polars as pl
    import sklearn as sk
    import cynde.functional as cf

@cv_stub.function(image=datascience_image, mounts=[modal.Mount.from_local_dir(LOCAL_MOUNT_PATH, remote_path="/root/cynde_mount")])
def fit_models_modal(models: Dict[str, List[Dict[str, Any]]], feature_name: str, indices_train:np.ndarray,indices_val: np.ndarray,indices_test:np.ndarray,
               fold_meta: Dict[str, Any],mount_directory:str) -> Tuple[List[pl.DataFrame], List[pl.DataFrame]]:

    
    pred_list = []
    results_list = []
    X,y = load_arrays_from_mount_modal(feature_name = feature_name)
    x_tr, y_tr = X[indices_train,:], y[indices_train]
    x_val, y_val = X[indices_val,:], y[indices_val]
    x_te, y_te = X[indices_test,:], y[indices_test]

    for model, hp_list in models.items():
        for hp in hp_list:
            pred_df_col, results_df_row = fit_clf_from_np_modal(x_tr, y_tr, x_val, y_val, x_te, y_te,
                                                                fold_metadata=fold_meta,
                                                  classifier=model,
                                                  classifier_hp=hp,
                                                  input_features_name=feature_name)
            pred_list.append(pred_df_col)
            
            fold_meta_data_df = pl.DataFrame({
                "k_outer": [fold_meta["k_outer"]],
                "k_inner": [fold_meta["k_inner"]],
                "r_outer": [fold_meta["r_outer"]],
                "r_inner": [fold_meta["r_inner"]]
            })
            
            results_df_row = pl.concat([results_df_row, fold_meta_data_df], how="horizontal") 
            results_list.append(results_df_row)
    return pred_list, results_list

def load_arrays_from_mount_local(feature_name:str,mount_directory:str):
        X = np.load(os.path.join(mount_directory,feature_name+".npy"))
        y = np.load(os.path.join(mount_directory,"labels.npy"))
        return X,y
    

def fit_models_local(models: Dict[str, List[Dict[str, Any]]], feature_name: str, indices_train:np.ndarray,indices_val: np.ndarray,indices_test:np.ndarray,
               fold_meta: Dict[str, Any],mount_directory:str) -> Tuple[List[pl.DataFrame], List[pl.DataFrame]]:

    
    pred_list = []
    results_list = []
    X,y = load_arrays_from_mount_local(feature_name = feature_name,mount_directory=mount_directory)
    x_tr, y_tr = X[indices_train,:], y[indices_train]
    x_val, y_val = X[indices_val,:], y[indices_val]
    x_te, y_te = X[indices_test,:], y[indices_test]

    for model, hp_list in models.items():
        for hp in hp_list:
            pred_df_col, results_df_row = fit_clf_from_np_modal(x_tr, y_tr, x_val, y_val, x_te, y_te,
                                                                fold_metadata=fold_meta,
                                                  classifier=model,
                                                  classifier_hp=hp,
                                                  input_features_name=feature_name)
            pred_list.append(pred_df_col)
            
            fold_meta_data_df = pl.DataFrame({
                "k_outer": [fold_meta["k_outer"]],
                "k_inner": [fold_meta["k_inner"]],
                "r_outer": [fold_meta["r_outer"]],
                "r_inner": [fold_meta["r_inner"]]
            })
            
            results_df_row = pl.concat([results_df_row, fold_meta_data_df], how="horizontal") 
            results_list.append(results_df_row)
    return pred_list, results_list

def preprocess_np_modal(df: pl.DataFrame,
                        mount_dir: str ,
                        inputs: List[Dict[str, Union[List[str], List[List[str]]]]],
                        target_column:str = "target") -> Tuple[pl.DataFrame, pl.DataFrame]:
    

    # Preprocess the dataset
    preprocess_start_time = time.time()
    feature_arrays, labels, _ = preprocess_dataset(df, inputs, target_column=target_column)
    #save the arrays to cynde_mount folder
    print(f"Saving arrays to {mount_dir}")
    for feature_name,feature_array in feature_arrays.items():
        np.save(os.path.join(mount_dir,feature_name+".npy"),feature_array)
    np.save(os.path.join(mount_dir,"labels.npy"),labels)
    preprocess_end_time = time.time()
    print(f"Preprocessing completed in {preprocess_end_time - preprocess_start_time} seconds")

def check_preprocessed_np_modal(mount_dir: str, inputs: List[Dict[str, Union[List[str], List[List[str]]]]]) -> bool:
    for feature_dict in inputs:
        for feature_list in feature_dict.values():
            for feature in feature_list:
                if not os.path.exists(os.path.join(mount_dir,feature+".npy")):
                    raise ValueError(f"Feature {feature} not found in {mount_dir}")
    return True

def train_nested_cv_from_np_modal(df: pl.DataFrame,
                        cv_type: Tuple[str, str],
                        mount_dir: str ,
                        inputs: List[Dict[str, Union[List[str], List[List[str]]]]],
                        models: Dict[str, List[Dict[str, Any]]],
                        group_outer: List[str],
                        k_outer: int,
                        group_inner: List[str],
                        k_inner: int,
                        r_outer: int = 1,
                        r_inner: int = 1,
                        run_local: bool = False,
                        target_column:str = "target",
                        load_preprocess:bool=True) -> Tuple[pl.DataFrame, pl.DataFrame]:
    start_time = time.time()
    df = check_add_cv_index(df)
    pred_df = nested_cv(df, cv_type, group_outer, k_outer, group_inner, k_inner, r_outer, r_inner, return_joined=False)
    cv_df = df.join(pred_df, on="cv_index", how="left")
    results_df = pl.DataFrame(schema=RESULTS_SCHEMA)
    print("results schema: ", results_df.schema)
    print(f"results shape: {results_df.shape}")

    # Preprocess the dataset
    preprocess_start_time = time.time()
    if load_preprocess:
        check_preprocessed_np_modal(mount_dir,inputs)
        feature_names = cf.derive_feature_names(inputs)
    else:
        feature_arrays, labels, _ = preprocess_dataset(df, inputs, target_column=target_column)
        #save the arrays to cynde_mount folder
        print(f"Saving arrays to {mount_dir}")
        for feature_name,feature_array in feature_arrays.items():
            np.save(os.path.join(mount_dir,feature_name+".npy"),feature_array)
        np.save(os.path.join(mount_dir,"labels.npy"),labels)
        feature_names = list(feature_arrays.keys())
    preprocess_end_time = time.time()
    print(f"Preprocessing completed in {preprocess_end_time - preprocess_start_time} seconds")

    all_pred_list = []
    all_results_list = []

    # Generate folds and fit models
    folds_generation_start_time = time.time()
    fit_tasks = generate_folds_from_np_modal_compatible(models,cv_df, cv_type, feature_names, group_outer, k_outer, group_inner, k_inner, r_outer, r_inner, mount_dir)
    if not run_local:
        all_tuples_list = fit_models_modal.starmap(fit_tasks)
    else:
        all_tuples_list = []
        for task in fit_tasks:
            all_tuples_list.append(fit_models_local(*task))

    #all_tuples_list is a list of tuples(list(pred_df),list(results_df)) we want to get to a single list
    for (pred_list,res_list) in all_tuples_list:
        all_results_list.extend(res_list)
        all_pred_list.extend(pred_list)
    #flatten the list of lists
    for frame in all_pred_list:
        pred_df = pred_df.join(frame, on="cv_index", how="left")
    folds_generation_end_time = time.time()
    print(f"Folds generation and model fitting completed in {folds_generation_end_time - folds_generation_start_time} seconds")
    print(f"Average time per fold: {(folds_generation_end_time - folds_generation_start_time) / (k_outer * k_inner * r_outer * r_inner)}")



    # Aggregate and save results
    aggregation_start_time = time.time()
    for results_df_row in all_results_list:
        results_df = results_df.vstack(results_df_row)


    aggregation_end_time = time.time()
    print(f"Aggregation of results completed in {aggregation_end_time - aggregation_start_time} seconds")

    save_results_start_time = time.time()
    # cf.save_results(df, results_df, pred_df, save_name, base_path=base_path)
    save_results_end_time = time.time()
    print(f"Saving results completed in {save_results_end_time - save_results_start_time} seconds")

    total_end_time = time.time()
    print(f"Total training and processing time: {total_end_time - start_time} seconds")
    tot_models =0
    for model,hp_list in models.items():
        tot_models += len(hp_list)

    print(f"Total average time per fold: {(total_end_time - start_time) / (k_outer * k_inner * r_outer * r_inner*tot_models*len(inputs))} seconds")

    return results_df, pred_df