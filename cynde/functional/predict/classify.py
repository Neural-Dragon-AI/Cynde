from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import polars as pl

from cynde.functional.predict.types import FeatureSet,InputConfig,ClassifierConfig,BaseClassifierConfig, LogisticRegressionConfig, RandomForestClassifierConfig, HistGradientBoostingClassifierConfig, CVConfig

def create_pipeline(df: pl.DataFrame, feature_set: FeatureSet, classifier_config: BaseClassifierConfig) -> Pipeline:
    """ maybne the df.schema is enough and we do not need to pass the whole df """
    transformers = []
    numerical_features = [feature.column_name for feature in feature_set.numerical]
    if numerical_features:
        scaler = feature_set.numerical[0].get_scaler()  # Assuming all numerical features use the same scaler
        transformers.append(("numerical", scaler, numerical_features))
    embedding_features = [feature.column_name for feature in feature_set.embeddings]
    if embedding_features:
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


def train_cv(df:pl.DataFrame, input_config:InputConfig, classifier_config:ClassifierConfig, cv_config:CVConfig) -> pl.DataFrame:
    """ Deploy a CV training pipeline to Modal, it requires a df with cv_index column and the features set to have already pre-processed and cached 
    1) Validate the input_config and check if the preprocessed features are present locally 
    2) create a generator that yields the modal path to the features and targets frames as well as the scikit pipeline object 
    3) execute through a modal starmap a script that fit end eval each pipeline on each feature set and return the results
    4) collect and aggregate the results locally and save and return the results
    """
