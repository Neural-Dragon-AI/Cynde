import logfire

logfire.install_auto_tracing(modules=['cynde'])
logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record='all'))
import os
import polars as pl
from typing import List, Optional, Tuple, Generator
import time
from cynde.functional.train.types import PredictConfig, BaseClassifierConfig,StratifiedConfig,Feature,FeatureSet,NumericalFeature, CategoricalFeature,EmbeddingFeature, InputConfig, ClassifierConfig, LogisticRegressionConfig, RandomForestClassifierConfig, HistGradientBoostingClassifierConfig, CVConfig, PipelineResults, PipelineInput, PurgedConfig
from cynde.functional.train.preprocess import convert_utf8_to_enum, check_add_cv_index

from sklearn.pipeline import Pipeline
from cynde.functional.train.train_modal import train_nested_cv_distributed
from cynde.functional.train.train_local import train_nested_cv
from cynde.functional.train.preprocess import preprocess_inputs, get_unique_columns


def load_tggate_grouped_data(data_path: str = r"C:\Users\Tommaso\Documents\Dev\Cynde\cache\tgca_nogen_simplified_smiles_malformer_embeddings_ada02_3large_3small_embeddings_grouped.parquet") -> pl.DataFrame:
    return pl.read_parquet(data_path)

df = load_tggate_grouped_data()
df = convert_utf8_to_enum(df, threshold=0.2)
df = check_add_cv_index(df,strict=False)

cols = ['COMPOUND_NAME',
        'SACRIFICE_PERIOD',
        'DOSE_LEVEL',
        'num_samples',
        'lesion_prob',
        'weight_mean',
        'SMILES_CODE_MoLFormer-XL-both-10pct_embeddings',
        'COMPOUND_NAME_text-embedding-ada-002_embeddings',
        'SMILES_CODE_text-embedding-ada-002_embeddings',
        'SACRIFICE_PERIOD_text-embedding-ada-002_embeddings',
        'DOSE_LEVEL_text-embedding-ada-002_embeddings',
        'DOSE_DOSE_UNIT_text-embedding-ada-002_embeddings',
        'COMPOUND_NAME_SACRIFICE_PERIOD_DOSE_LEVEL_text-embedding-ada-002_embeddings',
        'SMILES_CODE_SACRIFICE_PERIOD_DOSE_LEVEL_text-embedding-ada-002_embeddings',
        'COMPOUND_NAME_SMILES_CODE_SACRIFICE_PERIOD_DOSE_LEVEL_text-embedding-ada-002_embeddings',
        'COMPOUND_NAME_SACRIFICE_PERIOD_DOSE_DOSE_UNIT_text-embedding-ada-002_embeddings',
        'SMILES_CODE_SACRIFICE_PERIOD_DOSE_DOSE_UNIT_text-embedding-ada-002_embeddings',
        'COMPOUND_NAME_SMILES_CODE_SACRIFICE_PERIOD_DOSE_DOSE_UNIT_text-embedding-ada-002_embeddings',
        'COMPOUND_NAME_text-embedding-3-small_embeddings', 
        'SMILES_CODE_text-embedding-3-small_embeddings',
        'SACRIFICE_PERIOD_text-embedding-3-small_embeddings', 
        'DOSE_LEVEL_text-embedding-3-small_embeddings',
        'DOSE_DOSE_UNIT_text-embedding-3-small_embeddings', 
        'COMPOUND_NAME_DOSE_LEVEL_text-embedding-3-small_embeddings',
        'SMILES_CODE_DOSE_LEVEL_text-embedding-3-small_embeddings', 
        'COMPOUND_NAME_SMILES_CODE_DOSE_LEVEL_text-embedding-3-small_embeddings',
        'COMPOUND_NAME_DOSE_DOSE_UNIT_text-embedding-3-small_embeddings', 
        'SMILES_CODE_DOSE_DOSE_UNIT_text-embedding-3-small_embeddings',
        'COMPOUND_NAME_SMILES_CODE_DOSE_DOSE_UNIT_text-embedding-3-small_embeddings', 
        'COMPOUND_NAME_text-embedding-3-large_embeddings',
        'SMILES_CODE_text-embedding-3-large_embeddings', 
        'SACRIFICE_PERIOD_text-embedding-3-large_embeddings', 
        'DOSE_LEVEL_text-embedding-3-large_embeddings',
        'DOSE_DOSE_UNIT_text-embedding-3-large_embeddings', 
        'COMPOUND_NAME_DOSE_LEVEL_text-embedding-3-large_embeddings',
        'SMILES_CODE_DOSE_LEVEL_text-embedding-3-large_embeddings', 
        'COMPOUND_NAME_SMILES_CODE_DOSE_LEVEL_text-embedding-3-large_embeddings',
        'COMPOUND_NAME_DOSE_DOSE_UNIT_text-embedding-3-large_embeddings', 
        'SMILES_CODE_DOSE_DOSE_UNIT_text-embedding-3-large_embeddings', 
        'COMPOUND_NAME_SMILES_CODE_DOSE_DOSE_UNIT_text-embedding-3-large_embeddings', 
        'COMPOUND_NAME_SMILES_CODE_SACRIFICE_PERIOD_DOSE_DOSE_UNIT_text-embedding-3-large_embeddings', 
        'SMILES_CODE_SACRIFICE_PERIOD_DOSE_DOSE_UNIT_text-embedding-3-large_embeddings', 
        'COMPOUND_NAME_SACRIFICE_PERIOD_DOSE_DOSE_UNIT_text-embedding-3-large_embeddings', 
        'COMPOUND_NAME_SMILES_CODE_SACRIFICE_PERIOD_DOSE_LEVEL_text-embedding-3-large_embeddings', 
        'SMILES_CODE_SACRIFICE_PERIOD_DOSE_LEVEL_text-embedding-3-large_embeddings', 
        'COMPOUND_NAME_SACRIFICE_PERIOD_DOSE_LEVEL_text-embedding-3-large_embeddings', 
        'COMPOUND_NAME_SMILES_CODE_SACRIFICE_PERIOD_DOSE_DOSE_UNIT_text-embedding-3-small_embeddings', 
        'SMILES_CODE_SACRIFICE_PERIOD_DOSE_DOSE_UNIT_text-embedding-3-small_embeddings', 
        'COMPOUND_NAME_SACRIFICE_PERIOD_DOSE_DOSE_UNIT_text-embedding-3-small_embeddings', 
        'COMPOUND_NAME_SMILES_CODE_SACRIFICE_PERIOD_DOSE_LEVEL_text-embedding-3-small_embeddings', 
        'SMILES_CODE_SACRIFICE_PERIOD_DOSE_LEVEL_text-embedding-3-small_embeddings', 
        'COMPOUND_NAME_SACRIFICE_PERIOD_DOSE_LEVEL_text-embedding-3-small_embeddings', 
        'target']


#the best model combines compound_name, dose_level and sacrifice period embeddings each indipendently using hte large embeddings
feature_set_best_model = {"embeddings":[{"column_name":"COMPOUND_NAME_text-embedding-3-large_embeddings",
                                         "name":"embeddings of the compounds names",
                                         "embedder": "text-embedding-3-large_embeddings",
                                         "embedding_size":3072},
                                         {"column_name":"DOSE_LEVEL_text-embedding-3-large_embeddings",
                                            "name":"embeddings of the dose levels",
                                            "embedder": "text-embedding-3-large_embeddings",
                                            "embedding_size":3072},
                                            {"column_name":"SACRIFICE_PERIOD_text-embedding-3-large_embeddings",
                                            "name":"embeddings of the sacrifice periods",
                                            "embedder": "text-embedding-3-large_embeddings",
                                            "embedding_size":3072}]}

feature_set_only_compound_name = {"embeddings":[{"column_name":"COMPOUND_NAME_text-embedding-3-large_embeddings",
                                            "name":"embeddings of the compounds names",
                                            "embedder": "text-embedding-3-large_embeddings",
                                            "embedding_size":3072}]}

input_config_data = {"feature_sets":[feature_set_best_model,feature_set_only_compound_name],
                        "target_column":"target",
                        "save_folder":"C:/Users/Tommaso/Documents/Dev/Cynde/cynde_mount/",
                        "remote_folder":"/cynde_mount"}

classifiers_config = ClassifierConfig(classifiers=[RandomForestClassifierConfig(n_estimators=1000, max_depth=15)])

input_config = InputConfig.model_validate(input_config_data,context={"df":df})

inner_groups = ["DOSE_LEVEL","SACRIFICE_PERIOD","target"]
outer_groups = ["COMPOUND_NAME"]

cv_config = CVConfig(inner= StratifiedConfig(groups=inner_groups,k=5),
                     inner_replicas=1,
                     outer = PurgedConfig(groups=outer_groups,k=5),
                        outer_replicas=1)

task = PredictConfig(input_config=input_config, cv_config=cv_config, classifiers_config=classifiers_config)






df_filtered = df.select(get_unique_columns(task))

preprocess_inputs(df_filtered,task.input_config)
# results = train_nested_cv(df,task)

results = train_nested_cv_distributed(df_filtered,task)

#todo
# 1) fix the cv objects for purged (add a test) to the pydantic object I guess
# 3) results saving and aggregation