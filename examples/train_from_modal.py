# import logfire

# logfire.install_auto_tracing(modules=['cynde'])
# logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record='all'))
import os
import polars as pl
from typing import List, Optional, Tuple, Generator
import time
from cynde.functional.train.types import PredictConfig, BaseClassifierConfig,StratifiedConfig,Feature,FeatureSet,NumericalFeature, CategoricalFeature,EmbeddingFeature, InputConfig, ClassifierConfig, LogisticRegressionConfig, RandomForestClassifierConfig, HistGradientBoostingClassifierConfig, CVConfig, PipelineResults, PipelineInput
from cynde.functional.train.preprocess import convert_utf8_to_enum, check_add_cv_index

from sklearn.pipeline import Pipeline
import modal


from cynde.functional.train.train_modal import train_nested_cv_distributed

def load_minihermes_data(data_path: str = r"C:\Users\Tommaso\Documents\Dev\Cynde\cache\OpenHermes-2.5_embedded.parquet") -> pl.DataFrame:
    return pl.read_parquet(data_path)



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
                        "save_folder":"C:/Users/Tommaso/Documents/Dev/Cynde/cynde_mount/",
                        "remote_folder":"/cynde_mount"}



df = load_minihermes_data()
df = convert_utf8_to_enum(df, threshold=0.2)
df = check_add_cv_index(df,strict=False)

input_config = InputConfig.model_validate(input_config_data,context={"df":df})
# print("Input config:")
# print(input_config)


classifiers_config = ClassifierConfig(classifiers=[RandomForestClassifierConfig(n_estimators=100),RandomForestClassifierConfig(n_estimators=500)])
# print("Classifiers config:")
# print(classifiers_config)
groups = ["target"]
cv_config = CVConfig(inner= StratifiedConfig(groups=groups,k=5),
                     inner_replicas=1,
                     outer = StratifiedConfig(groups=groups,k=5),
                        outer_replicas=1)
# print("CV config:")
# print(cv_config)
# validate_preprocessed_inputs(input_config)

# print(df.columns)
task = PredictConfig(input_config=input_config, cv_config=cv_config, classifiers_config=classifiers_config)


results = train_nested_cv_distributed(df,task)
