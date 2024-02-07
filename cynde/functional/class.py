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


def get_features(df:pl.DataFrame,embedding_cols:list[str],categorical_cols:list[str]):
        print(f"Number of samples inside geat_feature: {df.shape[0]}")
        y_final = df['target'].to_numpy()
        print(f"Number of samples for the test set inside geat_feature: {y_final.shape[0]}")
        #get train embeddings
        embeddings = []
        if len(embedding_cols)>0:
            for col in embedding_cols:
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
        if len(categorical_cols)==0 and len(embedding_cols)>0:
            X_final = embeddings_np
        #case 2 only categorical
        elif len(categorical_cols)>0 and len(embedding_cols)==0:
            X_final = X_encoded
        #case 3 both
        elif len(categorical_cols)>0 and len(embedding_cols)>0:
            X_final = np.concatenate([embeddings_np, X_encoded], axis=1)
        else:
            raise ValueError("No features selected")
        return X_final,y_final

def fit_clf(X_train, y_train, X_val, y_val,X_test, y_test,classifier:str="RandomForest"):
    # Initialize the Random Forest classifier
    if classifier=="RandomForest":
        clf = RandomForestClassifier(random_state=777, n_estimators=1000, max_depth=15, n_jobs=-1)
    elif classifier == "NearestNeighbors":
        clf = KNeighborsClassifier(n_neighbors=7)
    elif classifier == "GaussianProcess":
        clf = GaussianProcessClassifier(kernel=RBF(1.0), random_state=42)
    elif classifier == "MLP":
        clf = MLPClassifier(alpha=1, max_iter=1000, random_state=42, hidden_layer_sizes=(1000, 500))
    clf = make_pipeline(StandardScaler(), clf)

    # Train the classifier using the training set
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred_val = clf.predict(X_val)
    y_pred_test = clf.predict(X_test)
    print("pred_shape",y_pred_test.shape)
    print("test_shape",y_test.shape)

    # Evaluate the classifier
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    mcc_test = matthews_corrcoef(y_test, y_pred_test)
    mcc_val = matthews_corrcoef(y_val, y_pred_val)
    classification_report_str = classification_report(y_test, y_pred_test)
    # print nicely formatted: accuracy, cv_scores.mean(), cv_scores.std(), 
    print(f"Accuracy Test: {accuracy_test}")
    print(f"Accuracy Val: {accuracy_val}")
    print(f"MCC Test: {mcc_test}")
    print(f"MCC Val: {mcc_val}")
    print(classification_report_str)
    return accuracy_test, mcc_test,classification_report_str

