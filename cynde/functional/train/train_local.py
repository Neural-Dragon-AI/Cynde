from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import polars as pl
from typing import Tuple
import time
from cynde.functional.train.types import PipelineResults,PredictConfig,PipelineInput,FeatureSet,InputConfig,ClassifierConfig,BaseClassifierConfig, LogisticRegressionConfig, RandomForestClassifierConfig, HistGradientBoostingClassifierConfig, CVConfig
from cynde.functional.train.cv import train_test_val,generate_nested_cv
from cynde.functional.train.preprocess import load_preprocessed_features,check_add_cv_index,validate_preprocessed_inputs

def create_pipeline(df: pl.DataFrame, feature_set: FeatureSet, classifier_config: BaseClassifierConfig) -> Pipeline:
    """ maybne the df.schema is enough and we do not need to pass the whole df """
    transformers = []
    numerical_features = [feature.column_name for feature in feature_set.numerical]
    if numerical_features:
        scaler = feature_set.numerical[0].get_scaler()  # Assuming all numerical features use the same scaler
        transformers.append(("numerical", scaler, numerical_features))
    embedding_features = [feature.column_name for feature in feature_set.embeddings]
    if embedding_features:
        #embedding features are stored as list[float] in polars but we map them to multiple columns of float in sklearn
        # so here we assume that we already pre-processed each embedding_feature to bea  lsit of columns of format column_name_{i}
        #accumulate for each embedding feature the list of columns that represent it and flatten it
        embedding_features = [f"{feature}_{i}" for feature in embedding_features for i in range(0,feature_set.embeddings[0].embedding_size)]
        scaler = feature_set.embeddings[0].get_scaler()  # Assuming all embedding features use the same scaler
        transformers.append(("embedding", scaler, embedding_features))

    categorical_features = [feature.column_name for feature in feature_set.categorical]
    if categorical_features:
        for feature in feature_set.categorical:
            if feature.one_hot_encoding:
                if df[feature.column_name].dtype == pl.Categorical:
                    categories = [df[feature.column_name].unique().to_list()]
                elif df[feature.column_name].dtype == pl.Enum:
                    categories = [df[feature.column_name].dtype.categories]
                else:
                    raise ValueError(f"Column '{feature.column_name}' must be of type pl.Categorical or pl.Enum for one-hot encoding.")
                one_hot_encoder = OneHotEncoder(categories=categories, handle_unknown='error', sparse_output=False)
                transformers.append((f"categorical_{feature.column_name}", one_hot_encoder, [feature.column_name]))
            else:
                if df[feature.column_name].dtype not in [pl.Float32, pl.Float64]:
                    raise ValueError(f"Column '{feature.column_name}' must be of type pl.Float32 or pl.Float64 for physical representation.")
                transformers.append((f"categorical_{feature.column_name}", "passthrough", [feature.column_name]))

    preprocessor = ColumnTransformer(transformers)

    # Create the classifier based on the classifier configuration
    if isinstance(classifier_config, LogisticRegressionConfig):
        classifier = LogisticRegression(**classifier_config.dict(exclude={"classifier_name"}))
    elif isinstance(classifier_config, RandomForestClassifierConfig):
        classifier = RandomForestClassifier(**classifier_config.dict(exclude={"classifier_name"}))
    elif isinstance(classifier_config, HistGradientBoostingClassifierConfig):
        classifier = HistGradientBoostingClassifier(**classifier_config.dict(exclude={"classifier_name"}))
    else:
        raise ValueError(f"Unsupported classifier: {classifier_config.classifier_name}")

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    pipeline.set_output(transform="polars")
    return pipeline

def evaluate_model(pipeline: Pipeline, X, y):
    """ Gotta make sure the returned predictions have the cv_index column"""
    predictions = pipeline.predict(X)
    accuracy = accuracy_score(y, predictions)
    mcc = matthews_corrcoef(y,predictions)
    pred_df = pl.DataFrame({"cv_index":X["cv_index"],"predictions":predictions})
    return pred_df,accuracy, mcc


def train_pipeline(input_config:InputConfig,pipeline_input:PipelineInput) -> Tuple[pl.DataFrame,pl.DataFrame,float,float]:
    feature_set = input_config.feature_sets[pipeline_input.feature_index]
    df_fold = load_preprocessed_features(input_config,pipeline_input.feature_index)
    print(df_fold)
    df_train,df_val,df_test = train_test_val(df_fold,pipeline_input.train_idx,pipeline_input.val_idx,pipeline_input.test_idx)
    print(df_train)
    pipeline = create_pipeline(df_train, feature_set, pipeline_input.cls_config)
    print(pipeline)
    pipeline.fit(df_train,df_train["target"])
    train_predictions, train_accuracy, train_mcc = evaluate_model(pipeline, df_train, df_train["target"])
    val_predictions,val_accuracy, val_mcc = evaluate_model(pipeline, df_val, df_val["target"])
    test_predictions,test_accuracy,test_mcc = evaluate_model(pipeline, df_test, df_test["target"])
    return PipelineResults(train_predictions = train_predictions,
                           val_predictions=val_predictions,
                           test_predictions=test_predictions,
                           train_accuracy=train_accuracy,
                           train_mcc=train_mcc,
                           val_accuracy=val_accuracy,
                           val_mcc=val_mcc,
                           test_accuracy=test_accuracy,
                           test_mcc=test_mcc,
                           pipeline_input=pipeline_input)




def train_nested_cv(df:pl.DataFrame, task_config:PredictConfig) -> pl.DataFrame:
    """ Deploy a CV training pipeline to Modal, it requires a df with cv_index column and the features set to have already pre-processed and cached 
    1) Validate the input_config and check if the preprocessed features are present locally 
    2) create a generator that yields the modal path to the features and targets frames as well as the scikit pipeline object 
    3) execute through a modal starmap a script that fit end eval each pipeline on each feature set and return the results
    4) collect and aggregate the results locally and save and return the results
    """
    #validate the inputs and check if the preprocessed features are present locally
    df = check_add_cv_index(df,strict=True)
    validate_preprocessed_inputs(task_config.input_config)
    
    #extract the subset of columns necessary for constructing the cross validation folds 
    unique_groups = list(set(task_config.cv_config.inner.groups + task_config.cv_config.outer.groups))
    df_idx = df.select(pl.col("cv_index"),pl.col(unique_groups))
    print(f"using groups {unique_groups} to generate the cv folds" )

    nested_cv = generate_nested_cv(df_idx,task_config)

    for pipeline_input in nested_cv:
        start = time.time()
        print(f"Training pipeline with classifier {pipeline_input.cls_config.classifier_name} on feature set {task_config.input_config.feature_sets[pipeline_input.feature_index]}")
        results = train_pipeline(task_config.input_config,pipeline_input)
        print(results)
        end = time.time()
        print(f"Training pipeline took {end-start} seconds")
