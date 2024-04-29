from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import polars as pl

from cynde.functional.predict.types import InputConfig, ClassifierConfig, LogisticRegressionConfig, RandomForestClassifierConfig, HistGradientBoostingClassifierConfig

def create_pipeline(df: pl.DataFrame, input_config: InputConfig, classifier_config: ClassifierConfig) -> Pipeline:
    """ maybne the df.schema is enough and we do not need to pass the whole df """
    transformers = []
    for feature_set in input_config.feature_sets:
        numerical_features = [feature.column_name for feature in feature_set.numerical]
        if numerical_features:
            scaler = feature_set.numerical[0].get_scaler()  # Assuming all numerical features use the same scaler
            transformers.append(("numerical", scaler, numerical_features))

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
    if isinstance(classifier_config.classifier, LogisticRegressionConfig):
        classifier = LogisticRegression(**classifier_config.classifier.dict(exclude={"classifier_name"}))
    elif isinstance(classifier_config.classifier, RandomForestClassifierConfig):
        classifier = RandomForestClassifier(**classifier_config.classifier.dict(exclude={"classifier_name"}))
    elif isinstance(classifier_config.classifier, HistGradientBoostingClassifierConfig):
        classifier = HistGradientBoostingClassifier(**classifier_config.classifier.dict(exclude={"classifier_name"}))
    else:
        raise ValueError(f"Unsupported classifier: {type(classifier_config.classifier)}")

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    pipeline.set_output(transform="polars")
    return pipeline
