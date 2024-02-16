from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
import polars as pl
import time
from typing import Tuple

def get_features(df:pl.DataFrame,numerical_cols:list[str],categorical_cols:list[str]):
        # print(f"Number of samples inside geat_feature: {df.shape[0]}")
        y_final = df['target'].to_numpy()
        # print(f"Number of samples for the test set inside geat_feature: {y_final.shape[0]}")
        #get train embeddings
        embeddings = []
        if len(numerical_cols)>0:
            for col in numerical_cols:
                embedding_np = np.array(df[col].to_list())
                embeddings.append(embedding_np)
            embeddings_np = np.concatenate(embeddings,axis=1)
        
        # Selecting only the categorical columns and the target
        if len(categorical_cols)>0:
            X = df.select(pl.col(categorical_cols))
            # One-hot encoding the categorical variables
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
        return X_final,y_final

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

