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
from typing import Tuple, Optional, Dict, List, Generator, Any, Union

stub = modal.Stub()


datascience_image = (
    Image.debian_slim(python_version="3.12.1")
    .pip_install("polars", "scikit-learn")
)

with datascience_image.imports():
    import polars as pl
    import sklearn as sk

def generate_folds_from_np_modal_compatible(cv_df: pl.DataFrame, cv_type: Tuple[str, str], feature_arrays: Dict[str, np.ndarray],
                           labels: np.ndarray, group_outer: List[str], k_outer: int, group_inner: List[str], k_inner: int,
                           r_outer: int, r_inner: int) -> Generator:
    for r_o in range(r_outer):
        for k_o in range(k_outer):
            for r_i in range(r_inner):
                for k_i in range(k_inner):
                    fold_name = cf.get_fold_name_cv(group_outer, cv_type, r_o, k_o, group_inner, r_i, k_i)
                    indices_train, indices_val, indices_test = cf.fold_to_indices(cv_df, fold_name)

                    for feature_name, X in feature_arrays.items():
                        y = labels
                        x_tr, y_tr = X[indices_train,:], y[indices_train]
                        x_val, y_val = X[indices_val,:], y[indices_val]
                        x_te, y_te = X[indices_test,:], y[indices_test]

                        fold_meta = {
                            "r_outer": r_o,
                            "k_outer": k_o,
                            "r_inner": r_i,
                            "k_inner": k_i,
                            "fold_name": fold_name,
                            "train_index": pl.Series(indices_train),
                            "val_index": pl.Series(indices_val),
                            "test_index": pl.Series(indices_test),
                        }

                        yield (x_tr, y_tr, x_val, y_val, x_te, y_te, fold_meta, feature_name)

@stub.function(image=datascience_image)
def fit_clf_from_np_modal(X_train, y_train, X_val, y_val, X_test, y_test,fold_metadata:dict,
            classifier: str = "RandomForest", classifier_hp: dict = {}, input_features_name: str = "") -> Tuple[pl.DataFrame, pl.DataFrame]:
    
    return cf.fit_clf_from_np(X_train, y_train, X_val, y_val, X_test, y_test,fold_metadata,
            classifier, classifier_hp, input_features_name)


def train_nested_cv_from_np(df: pl.DataFrame,
                           cv_type: Tuple[str, str],
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
    df = cf.check_add_cv_index(df)
    pred_df = cf.nested_cv(df, cv_type, group_outer, k_outer, group_inner, k_inner, r_outer, r_inner, return_joined=False)
    cv_df = df.join(pred_df, on="cv_index", how="left")
    results_df = pl.DataFrame(schema=cf.RESULTS_SCHEMA)
    print("results schema: ", results_df.schema)
    print(f"results shape: {results_df.shape}")

    # Preprocess the dataset
    preprocess_start_time = time.time()
    feature_arrays, labels, _ = cf.preprocess_dataset(df, inputs)
    preprocess_end_time = time.time()
    print(f"Preprocessing completed in {preprocess_end_time - preprocess_start_time} seconds")

    all_pred_list = []
    all_results_list = []

    # Generate folds and fit models
    folds_generation_start_time = time.time()
    fit_tasks = generate_folds_from_np_modal_compatible(cv_df, cv_type, feature_arrays, labels, group_outer, k_outer, group_inner, k_inner, r_outer, r_inner)
    if not skip_class:
        fit_results = fit_clf_from_np_modal.starmap(fit_tasks)

        all_pred_list = [pred_df for pred_df, _ in fit_results]
        all_results_list = [results_df for _, results_df in fit_results]
    folds_generation_end_time = time.time()
    print(f"Folds generation and model fitting completed in {folds_generation_end_time - folds_generation_start_time} seconds")
    print(f"Average time per fold: {(folds_generation_end_time - folds_generation_start_time) / (k_outer * k_inner * r_outer * r_inner)}")


    # Aggregate and save results
    aggregation_start_time = time.time()
    for results_df_row in all_results_list:
        results_df = results_df.vstack(results_df_row)

    for pred_df_col in all_pred_list:
        pred_df = pred_df.join(pred_df_col, on="cv_index", how="left")
    aggregation_end_time = time.time()
    print(f"Aggregation of results completed in {aggregation_end_time - aggregation_start_time} seconds")

    save_results_start_time = time.time()
    cf.save_results(df, results_df, pred_df, save_name, base_path=base_path)
    save_results_end_time = time.time()
    print(f"Saving results completed in {save_results_end_time - save_results_start_time} seconds")

    total_end_time = time.time()
    print(f"Total training and processing time: {total_end_time - start_time} seconds")
    print(f"Total average time per fold: {(total_end_time - start_time) / (k_outer * k_inner * r_outer * r_inner)} seconds")

    return results_df, pred_df

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

    # Call the train_nested_cv_from_np function with the required arguments
    results_df, pred_df = train_nested_cv_from_np(embedded_df,
                                                 ("vanilla", "kfold"),
                                                 inputs,
                                                 models_dict,
                                                 group_outer=None,
                                                 k_outer=5,
                                                 group_inner=None,
                                                 k_inner=5,
                                                 r_outer=1,
                                                 r_inner=1,
                                                 save_name="nested_cv_out",
                                                 base_path=cache_dir,
                                                 skip_class=False)

    print("Training completed successfully!")
    summary = cf.results_summary(results_df,by_test_fold=True)
    print(summary)




    