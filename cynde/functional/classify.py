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
from typing import Tuple, Optional, List, Dict, Union, Any
import os

def get_hp_classifier_name(classifier_hp:dict) -> str:
    classifier_hp_name = "_".join([f"{key}_{value}" for key,value in classifier_hp.items()])
    return classifier_hp_name

def get_pred_column_name(fold_name:str,input_features_name:str,classifier:str,classifier_hp_name:str) -> str:
    return  "{}_{}_{}_{}_y_pred".format(fold_name,input_features_name,classifier,classifier_hp_name)

def get_input_name(input_feature:Dict):
    numerical_cols = input_feature.get("numerical", []) + input_feature.get("embeddings", [])
    categorical_cols = input_feature.get("categorical", [])
    feature_name = "_".join(numerical_cols + categorical_cols)
    return feature_name

def fold_to_indices(fold_frame: pl.DataFrame, fold_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts indices for training, validation, and test sets based on fold configuration.

    Parameters:
    - fold_frame: DataFrame with columns ['cv_index', '<fold_name>'] where '<fold_name>' column contains
      identifiers for train, validation, and test sets.
    - fold_name: The name of the column in fold_frame that contains the fold identifiers.

    Returns:
    - Tuple of numpy arrays: (train_indices, validation_indices, test_indices)
    """
    # Assuming the fold_frame has a column with the name in fold_name that marks rows as 'train', 'val', or 'test'
    train_indices = fold_frame.filter(pl.col(fold_name) == "train").select("cv_index").to_numpy().flatten()
    val_indices = fold_frame.filter(pl.col(fold_name) == "val").select("cv_index").to_numpy().flatten()
    test_indices = fold_frame.filter(pl.col(fold_name) == "test").select("cv_index").to_numpy().flatten()

    return train_indices, val_indices, test_indices


def preprocess_dataset(df: pl.DataFrame, inputs: List[Dict[str, Union[List[str], List[List[str]]]]],target_column:str="target"):
    feature_arrays = {}
    encoders = {}

    for inp in inputs:
        numerical_cols = inp.get("numerical", []) + inp.get("embeddings", [])
        categorical_cols = inp.get("categorical", [])
        feature_name = "_".join(numerical_cols + categorical_cols)

        X_final,_ = get_features(df, numerical_cols, categorical_cols,target_column = target_column)
        
        feature_arrays[feature_name] = X_final
        print(f"Feature array shape for {feature_name}: {X_final.shape}")

    # Assuming 'target' is the label column
    labels = df[target_column].to_numpy()
    return feature_arrays, labels, encoders

def derive_feature_names(inputs: List[Dict[str, Union[List[str], List[List[str]]]]]) -> List[str]:
    feature_names = []
    for inp in inputs:
        numerical_cols = inp.get("numerical", []) + inp.get("embeddings", [])
        categorical_cols = inp.get("categorical", [])
        feature_name = "_".join(numerical_cols + categorical_cols)
        feature_names.append(feature_name)
    return feature_names


def get_features(df:pl.DataFrame,numerical_cols:list[str],categorical_cols:list[str],return_encoder:Optional[bool] = False,cat_encoder:Optional[OneHotEncoder]=None, target_column:str="target") -> Tuple[np.ndarray,np.ndarray]:
        # print(f"Number of samples inside geat_feature: {df.shape[0]}")
        y_final = df[target_column].to_numpy()
        # print(f"Number of samples for the test set inside geat_feature: {y_final.shape[0]}")
        #get train embeddings
        embeddings = []
        if len(numerical_cols)>0:
            for col in numerical_cols:
                embedding_np = np.array(df[col].to_list())
                embeddings.append(embedding_np)
            embeddings_np = np.concatenate(embeddings,axis=1)
        
        # Selecting only the categorical columns and the target
        encoder = None
        if len(categorical_cols)>0:
            X = df.select(pl.col(categorical_cols))
            # One-hot encoding the categorical variables
            if cat_encoder is None:
                encoder = OneHotEncoder()
            X_encoded = encoder.fit_transform(X)
            X_encoded = X_encoded.toarray()
        #case 1 only embeddings
        if len(categorical_cols)==0 and len(numerical_cols)>0:
            X_final = embeddings_np
        #case 2 only categorical
        elif len(categorical_cols)>0 and len(numerical_cols)==0:
            X_final = X_encoded
        #case 3 both
        elif len(categorical_cols)>0 and len(numerical_cols)>0:
            X_final = np.concatenate([embeddings_np, X_encoded], axis=1)
        else:
            raise ValueError("No features selected")
        if return_encoder:
           
            return X_final,y_final,encoder
        return X_final,y_final

def fit_clf(X_train, y_train, X_val, y_val,X_test, y_test,fold_frame:pl.DataFrame,classifier:str="RandomForest",classifier_hp:dict={},input_features_name:str="") -> Tuple[pl.DataFrame,pl.DataFrame]:
    start_time = time.time()
    # Initialize the Random Forest classifier
    if classifier=="RandomForest":
        if classifier_hp is None:
            classifier_hp = {"random_state":777,"n_estimators":100,"max_depth":5,"n_jobs":-1}
        clf = RandomForestClassifier(**classifier_hp)
        
    elif classifier == "NearestNeighbors":
        if classifier_hp is None:
            classifier_hp = {"n_neighbors":7}
        clf = KNeighborsClassifier(**classifier_hp)
    elif classifier == "MLP":
        if classifier_hp is None:
            classifier_hp = {"alpha":1, "max_iter":1000, "random_state":42, "hidden_layer_sizes":(1000, 500)}
        clf = MLPClassifier(**classifier_hp)
    #create the classifier_hp_name from the dictionary
    classifier_hp_name = "_".join([f"{key}_{value}" for key,value in classifier_hp.items()])
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
    # print(f"Accuracy Test: {accuracy_test}")
    # print(f"Accuracy Val: {accuracy_val}")
    # print(f"MCC Test: {mcc_test}")
    # print(f"MCC Val: {mcc_val}")
    # print(f"Total CLS time: {human_readable_total_time}")
    return pred_df,results_df

def fit_clf_from_np(X_train, y_train, X_val, y_val, X_test, y_test,fold_metadata:dict,
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
    classifier_hp_name = get_hp_classifier_name(classifier_hp)
    fold_name = fold_metadata["fold_name"]
    # pred_column_name = "{}_{}_{}_{}_y_pred".format(fold_name,input_features_name,classifier,classifier_hp_name)
    pred_column_name = get_pred_column_name(fold_name,input_features_name,classifier,classifier_hp_name)
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
    classifier_hp_name = get_hp_classifier_name(classifier_hp)
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