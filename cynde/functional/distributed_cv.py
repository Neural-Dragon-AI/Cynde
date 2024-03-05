from modal import Image
import modal
import polars as pl
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,  matthews_corrcoef
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import polars as pl
import time
from typing import Tuple, Optional, Dict, List, Generator, Any, Union
from cynde.functional.cv import generate_folds_from_np_modal_compatible, check_add_cv_index, preprocess_dataset, nested_cv, RESULTS_SCHEMA

cv_stub = modal.Stub("distributed_cv")

datascience_image = (
    Image.debian_slim(python_version="3.12.1")
    .apt_install("git")
    .pip_install("polars","scikit-learn","openai","tiktoken")#, force_build=True)
    .run_commands("git clone https://github.com/Neural-Dragon-AI/Cynde/")
    .run_commands("cd Cynde && pip install -r requirements.txt && pip install .")
)
with datascience_image.imports():
    import polars as pl
    import sklearn as sk
    import cynde.functional as cf

@cv_stub.function(image=datascience_image, mounts=[modal.Mount.from_local_dir(r"/Users/tommasofurlanello/Documents/Dev/Cynde/cynde_mount", remote_path="/root/cynde_mount")])
def fit_models_modal(models: Dict[str, List[Dict[str, Any]]], feature_name: str, indices_train:np.ndarray,indices_val: np.ndarray,indices_test:np.ndarray,
               fold_meta: Dict[str, Any],mount_directory:str) -> Tuple[List[pl.DataFrame], List[pl.DataFrame]]:
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score,  matthews_corrcoef
    from sklearn.preprocessing import OneHotEncoder
    import time
    def load_arrays_from_mount_modal(feature_name:str):
        X = np.load(os.path.join("/root/cynde_mount",feature_name+".npy"))
        y = np.load(os.path.join("/root/cynde_mount","labels.npy"))
        return X,y
    def fit_clf_from_np_modal(X_train, y_train, X_val, y_val, X_test, y_test,fold_metadata:dict,
            classifier: str = "RandomForest", classifier_hp: dict = {}, input_features_name: str = "") -> Tuple[pl.DataFrame, pl.DataFrame]:
    
        start_time = time.time()
        clf = None
        #create classifiers 
        if classifier == "RandomForest":
            clf = RandomForestClassifier(**(classifier_hp or {"random_state": 777, "n_estimators": 100, "max_depth": 5, "n_jobs": -1}))
        elif classifier == "NearestNeighbors":
            clf = KNeighborsClassifier(**(classifier_hp or {"n_neighbors": 7}))
        elif classifier == "MLP":
            clf = MLPClassifier(**(classifier_hp or {"alpha": 1, "max_iter": 1000, "random_state": 42, "hidden_layer_sizes": (1000, 500)}))
        else:
            raise ValueError("Classifier not supported")
        
        #create names 
        classifier_hp_name = "_".join([f"{key}_{value}" for key,value in classifier_hp.items()])
        fold_name = fold_metadata["fold_name"]
        pred_column_name = "{}_{}_{}_{}_y_pred".format(fold_name,input_features_name,classifier,classifier_hp_name)

        # Train the classifier using the training set
        start_train_time = time.time()
        clf = make_pipeline(StandardScaler(), clf)
        # print("before training")
        # print("X_train shape: ", X_train.shape)
        # print("y_train shape: ", y_train.shape)
        clf.fit(X_train, y_train)
        end_train_time = time.time()
        human_readable_train_time = time.strftime("%H:%M:%S", time.gmtime(end_train_time-start_train_time))

        # Predictions
        start_pred_time = time.time()
        # print("before clf predict")
        y_pred_train = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)
        y_pred_test = clf.predict(X_test)
        end_pred_time = time.time()
        human_readable_pred_time = time.strftime("%H:%M:%S", time.gmtime(end_pred_time-start_pred_time))

        # Evaluation
        start_eval_time = time.time()
        # print("before evaluation")
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_val = accuracy_score(y_val, y_pred_val)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        mcc_train = matthews_corrcoef(y_train, y_pred_train) 
        mcc_val = matthews_corrcoef(y_val, y_pred_val) 
        mcc_test = matthews_corrcoef(y_test, y_pred_test) 
        end_eval_time = time.time()
        human_readable_eval_time = time.strftime("%H:%M:%S", time.gmtime(end_eval_time-start_eval_time))
        end_time = time.time()
        human_readable_total_time = time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))

        #package predictions
        pred_train_df = pl.DataFrame({"cv_index":fold_metadata["train_index"],pred_column_name:y_pred_train})
        pred_val_df = pl.DataFrame({"cv_index":fold_metadata["val_index"],pred_column_name:y_pred_val})
        pred_test_df = pl.DataFrame({"cv_index":fold_metadata["test_index"],pred_column_name:y_pred_test})
        pred_df = pred_train_df.vstack(pred_val_df).vstack(pred_test_df)

        #package results

        time_dict = {"train_time":human_readable_train_time,
                    "pred_time":human_readable_pred_time,
                    "eval_time":human_readable_eval_time,"total_cls_time":human_readable_total_time}

        results_df = pl.DataFrame({"classifier":[classifier],
                                "classifier_hp":[classifier_hp_name],
                                "fold_name":[fold_name],
                                "pred_name":[pred_column_name],
                                "input_features_name":[input_features_name],
                                "accuracy_train":[accuracy_train],
                                    "accuracy_val":[accuracy_val],
                                    "accuracy_test":[accuracy_test],
                                    "mcc_train":[mcc_train],
                                    "mcc_val":[mcc_val],
                                    "mcc_test":[mcc_test],
                                    "train_index":[fold_metadata["train_index"]],
                                    "val_index":[fold_metadata["val_index"]],
                                    "test_index":[fold_metadata["test_index"]],
                                    "time":[time_dict],
                                    }).unnest("time")
        

        return pred_df, results_df
    
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
                        save_name: str = "nested_cv_out",
                       
                        base_path: Optional[str] = None,
                        skip_class: bool = False) -> Tuple[pl.DataFrame, pl.DataFrame]:
    start_time = time.time()
    df = check_add_cv_index(df)
    pred_df = nested_cv(df, cv_type, group_outer, k_outer, group_inner, k_inner, r_outer, r_inner, return_joined=False)
    cv_df = df.join(pred_df, on="cv_index", how="left")
    results_df = pl.DataFrame(schema=RESULTS_SCHEMA)
    print("results schema: ", results_df.schema)
    print(f"results shape: {results_df.shape}")

    # Preprocess the dataset
    preprocess_start_time = time.time()
    feature_arrays, labels, _ = preprocess_dataset(df, inputs)
    #save the arrays to cynde_mount folder
    for feature_name,feature_array in feature_arrays.items():
        np.save(os.path.join(mount_dir,feature_name+".npy"),feature_array)
    np.save(os.path.join(mount_dir,"labels.npy"),labels)
    preprocess_end_time = time.time()
    print(f"Preprocessing completed in {preprocess_end_time - preprocess_start_time} seconds")

    all_pred_list = []
    all_results_list = []

    # Generate folds and fit models
    folds_generation_start_time = time.time()
    fit_tasks = generate_folds_from_np_modal_compatible(models,cv_df, cv_type, feature_arrays, group_outer, k_outer, group_inner, k_inner, r_outer, r_inner, mount_dir)
    if not skip_class:
        all_tuples_list = fit_models_modal.starmap(fit_tasks)
    #all_tuples_list is a list of tuples(list(pred_df),list(results_df)) we want to get to a single list
    for (pred_list,res_list) in all_tuples_list:
        all_results_list.extend(res_list)
        all_pred_list.extend(pred_list)
    #flatten the list of lists
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