import logfire

logfire.install_auto_tracing(modules=['cynde'])
logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record='all'))



from cynde.functional.distributed_cv import train_nested_cv_from_np_modal, cv_stub, preprocess_np_modal
import cynde.functional as cf
import os
import polars as pl
from polars.polars import PyExpr
from typing import List, Optional, Tuple, Generator
import time
from cynde.functional.predict.types import PredictConfig, BaseClassifierConfig,StratifiedConfig,Feature,FeatureSet,NumericalFeature, CategoricalFeature,EmbeddingFeature, InputConfig, ClassifierConfig, LogisticRegressionConfig, RandomForestClassifierConfig, HistGradientBoostingClassifierConfig, CVConfig
from cynde.functional.predict.preprocess import convert_utf8_to_enum, check_add_cv_index, preprocess_inputs, load_preprocessed_features
from cynde.functional.predict.cv import stratified_combinatorial
from cynde.functional.predict.classify import create_pipeline ,train_nested_cv

from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


def load_minihermes_data(data_path: str = r"C:\Users\Tommaso\Documents\Dev\Cynde\cache\OpenHermes-2.5_embedded.parquet") -> pl.DataFrame:
    return pl.read_parquet(data_path)

df = load_minihermes_data()
df = convert_utf8_to_enum(df, threshold=0.2)
df = check_add_cv_index(df,strict=False)
print(df.columns)

feature_set_small_data = {"embeddings":[{"column_name":"conversations_text-embedding-3-small_embeddings",
                                         "name":"feature set for the smaller oai embeddings",
                                         "embedder": "text-embedding-3-small_embeddings",
                                         "embedding_size":1536}]}
feature_set_large_data = {"embeddings":[{"column_name":"conversations_text-embedding-3-large_embeddings",
                                         "name":"feature set for the larger oai embeddings",
                                        "embedder": "text-embedding-3-large_embeddings",
                                         "embedding_size":3072}]}

input_config_data = {"feature_sets":[feature_set_small_data,feature_set_large_data],
                        "target_column":"target",
                        "save_folder":"C:/Users/Tommaso/Documents/Dev/Cynde/cynde_mount/"}

input_config = InputConfig.model_validate(input_config_data,context={"df":df})
print("Input config:")
print(input_config)
preprocess_inputs(df, input_config)

classifiers_config = ClassifierConfig(classifiers=[RandomForestClassifierConfig(n_estimators=100),RandomForestClassifierConfig(n_estimators=500)])
print("Classifiers config:")
print(classifiers_config)
groups = ["target"]
cv_config = CVConfig(inner= StratifiedConfig(groups=groups,k=5),
                     inner_replicas=1,
                     outer = StratifiedConfig(groups=groups,k=5),
                        outer_replicas=1)
print("CV config:")
print(cv_config)

task = PredictConfig(input_config=input_config, cv_config=cv_config, classifiers_config=classifiers_config)

train_nested_cv(df,task)

