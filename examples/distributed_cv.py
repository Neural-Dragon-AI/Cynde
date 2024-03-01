from modal import Image
import modal
from datasets import load_dataset
import polars as pl
import os
import cynde.functional as cf
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
from typing import Tuple, Optional

stub = modal.Stub()


datascience_image = (
    Image.debian_slim(python_version="3.12.1")
    .pip_install("polars", "scikit-learn")
)

with datascience_image.imports():
    import polars as pl
    import sklearn as sk

@stub.function(image=datascience_image)
def create_df(content:dict):
    df = pl.DataFrame(content)
    return df

@stub.function(image=datascience_image)
def fit_clf(X_train, y_train, X_val, y_val,X_test, y_test,fold_frame:pl.DataFrame,classifier:str="RandomForest",classifer_hp:dict={},input_features_name:str="") -> Tuple[pl.DataFrame,pl.DataFrame]:
    start_time = time.time()
    # Initialize the Random Forest classifier
    if classifier=="RandomForest":
        if classifer_hp is None:
            classifer_hp = {"random_state":777,"n_estimators":100,"max_depth":5,"n_jobs":-1}
        clf = RandomForestClassifier(**classifer_hp)
        
    elif classifier == "NearestNeighbors":
        if classifer_hp is None:
            classifer_hp = {"n_neighbors":7}
        clf = KNeighborsClassifier(**classifer_hp)
    elif classifier == "MLP":
        if classifer_hp is None:
            classifer_hp = {"alpha":1, "max_iter":1000, "random_state":42, "hidden_layer_sizes":(1000, 500)}
        clf = MLPClassifier(**classifer_hp)
    #create the classifier_hp_name from the dictionary
    classifier_hp_name = "_".join([f"{key}_{value}" for key,value in classifer_hp.items()])
    fold_name = fold_frame.columns[1]
    pred_column_name = "{}_{}_{}_{}_y_pred".format(fold_name,input_features_name,classifier,classifier_hp_name)
    clf = make_pipeline(StandardScaler(), clf)
    train_index_series = fold_frame.filter(pl.col(fold_name)=="train")["cv_index"]
    val_index_series = fold_frame.filter(pl.col(fold_name)=="val")["cv_index"]
    test_index_series = fold_frame.filter(pl.col(fold_name)=="test")["cv_index"]

    # Train the classifier using the training set
    start_train_time = time.time()
    clf.fit(X_train, y_train)
    # Predict on all the folds
    end_train_time = time.time()
    human_readable_train_time = time.strftime("%H:%M:%S", time.gmtime(end_train_time-start_train_time))
    start_pred_time = time.time()
    y_pred_train = clf.predict(X_train)
    y_pred_val = clf.predict(X_val)
    y_pred_test = clf.predict(X_test)
   

    pred_train_df = pl.DataFrame({"cv_index":train_index_series,pred_column_name:y_pred_train})
    pred_val_df = pl.DataFrame({"cv_index":val_index_series,pred_column_name:y_pred_val})
    pred_test_df = pl.DataFrame({"cv_index":test_index_series,pred_column_name:y_pred_test})
    pred_df = pred_train_df.vstack(pred_val_df).vstack(pred_test_df)

    end_pred_time = time.time()
    human_readable_pred_time = time.strftime("%H:%M:%S", time.gmtime(end_pred_time-start_pred_time))

    
    start_eval_time = time.time()
    # Evaluate the classifier
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    mcc_train = matthews_corrcoef(y_train, y_pred_train)
    mcc_test = matthews_corrcoef(y_test, y_pred_test)
    mcc_val = matthews_corrcoef(y_val, y_pred_val)
    end_eval_time = time.time()
    human_readable_eval_time = time.strftime("%H:%M:%S", time.gmtime(end_eval_time-start_eval_time))
    end_time = time.time()
    human_readable_total_time = time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))
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
                                "train_index":[train_index_series],
                                "val_index":[val_index_series],
                                "test_index":[test_index_series],
                                "time":[time_dict],
                                }).unnest("time")
    # print nicely formatted: accuracy, cv_scores.mean(), cv_scores.std(), 
    print(f"Accuracy Test: {accuracy_test}")
    print(f"Accuracy Val: {accuracy_val}")
    print(f"MCC Test: {mcc_test}")
    print(f"MCC Val: {mcc_val}")
    print(f"Total CLS time: {human_readable_total_time}")
    return pred_df,results_df



@stub.local_entrypoint()
def main():


    # Get the directory above the current directory
    above_dir = os.path.dirname(os.getcwd())

    # Specify the cache directory as 'data' subdirectory within the current directory
    cynde_dir = os.path.join(above_dir, "Cynde")

    cache_dir = os.path.join(cynde_dir, "cache")
    print(cache_dir)

    dataset_name = "OpenHermes-2.5"
    dataset_path = os.path.join(cache_dir, dataset_name)


    df = pl.read_parquet(dataset_path+"_filtered.parquet")
    print(df.head(1))
    print(df["source"].value_counts())

    df = df.with_columns(pl.col("conversations").list.eval(pl.element().struct.json_encode()).list.join("\n"),(pl.col("source")=="caseus_custom").alias("target"))

    #check if the embedded df already exist at dataset_path+"_embedded.parquet"
    if not os.path.exists(dataset_path+"_embedded_small_large.parquet"):
        embedded_df = cf.embed_columns(df, ["conversations"],models=["text-embedding-3-small","text-embedding-3-large"])
        embedded_df.write_parquet(dataset_path+"_embedded_small_large.parquet")
    else:
        embedded_df = pl.read_parquet(dataset_path+"_embedded_small_large.parquet")

    print(df["target"].value_counts())

    print(cf.vanilla_kfold(embedded_df,group=None,k=5))


    models_dict = {"RandomForest": [{"n_estimators": 10, "max_depth": 5},{"n_estimators": 50, "max_depth": 10}]}
    inputs =[{"numerical":["conversations_text-embedding-3-small_embeddings"]},
            {"numerical":["conversations_text-embedding-3-large_embeddings"]},
            {"numerical":["conversations_text-embedding-3-small_embeddings","conversations_text-embedding-3-large_embeddings"]}]
    inputs = [inputs[0]]



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
    #retrive the path of teh cynde package
    up_one = os.path.join(os.getcwd(),"data_processed")
    save_name_res = os.path.join(up_one,save_name_res)
    save_name_pred = os.path.join(up_one,save_name_pred)
    save_name_joined_df = os.path.join(up_one,save_name_joined_df)
    print(f"Saving results to {save_name_res}")
    results_df.write_parquet(save_name_res)
    pred_df.write_parquet(save_name_pred)
    df.join(pred_df, on="cv_index", how="left").write_parquet(save_name_joined_df)
    return results_df,pred_df

    for result in create_df.map(example_data):
        print(result)


