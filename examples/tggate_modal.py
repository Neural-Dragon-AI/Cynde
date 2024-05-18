# import logfire

# logfire.install_auto_tracing(modules=['cynde'])
# logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record='all'))
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
from cynde.functional.train.results import merge_results, merge_metrics,aggr_metrics_by_inputs, save_results


def load_tggate_grouped_data(data_path: str = r"C:\Users\Tommaso\Documents\Dev\Cynde\cache\tgca_nogen_simplified_smiles_malformer_embeddings_ada02_3large_3small_embeddings_grouped.parquet") -> pl.DataFrame:
    return pl.read_parquet(data_path)
def load_moldes_data(data_path:str = r"C:\Users\Tommaso\Documents\Dev\Cynde\cache\MolDes_3D_TG.tsv") -> pl.DataFrame:
    return pl.read_csv(data_path, separator="\t")

def load_and_prepare_data(data_path_tggate: str = r"C:\Users\Tommaso\Documents\Dev\Cynde\cache\tgca_nogen_simplified_smiles_malformer_embeddings_ada02_3large_3small_embeddings_grouped.parquet",
                           data_path_moldes:str = r"C:\Users\Tommaso\Documents\Dev\Cynde\cache\MolDes_3D_TG.tsv") -> pl.DataFrame:
    dft = load_tggate_grouped_data(data_path_tggate)
    dfm = load_moldes_data(data_path_moldes)
    dfm = dfm.with_columns(pl.lit(True).alias("in_moldes"))
    meta_cols = [col for col in dft.columns if "embeddings" not in col]
    text_embedding_cols = [col for col in dft.columns if "embeddings" in col]
    moldes_columns = [col for col in dfm.columns if col not in dft.columns]
    dfj = dft.join(dfm, on="COMPOUND_NAME", how="left")
    dfj = dfj.filter(~pl.col("in_moldes").is_null())
    dfjj = dfj.select(pl.col(meta_cols),pl.col(text_embedding_cols),pl.concat_list(moldes_columns).alias("COMPOUND_NAME_moldes_embeddings"))
    dfjj = (convert_utf8_to_enum(dfjj, threshold=0.2))
    return check_add_cv_index(dfjj,strict=False)

df = load_and_prepare_data()




feature_set_molformer={"embeddings":[{"column_name":"SMILES_CODE_MoLFormer-XL-both-10pct_embeddings",
                                        "name":"embeddings of the compound names",
                                        "embedder": "molformer",
                                        "embedding_size":768},
                                         {"column_name":"DOSE_LEVEL_text-embedding-3-large_embeddings",
                                            "name":"embeddings of the dose levels",
                                            "embedder": "text-embedding-3-large_embeddings",
                                            "embedding_size":3072},
                                            {"column_name":"SACRIFICE_PERIOD_text-embedding-3-large_embeddings",
                                            "name":"embeddings of the sacrifice periods",
                                            "embedder": "text-embedding-3-large_embeddings",
                                            "embedding_size":3072}]}

feature_set_moldes={"embeddings":[{"column_name":"COMPOUND_NAME_moldes_embeddings",
                                        "name":"embeddings of the compound names",
                                        "embedder": "molder",
                                        "embedding_size":1826},
                                         {"column_name":"DOSE_LEVEL_text-embedding-3-large_embeddings",
                                            "name":"embeddings of the dose levels",
                                            "embedder": "text-embedding-3-large_embeddings",
                                            "embedding_size":3072},
                                            {"column_name":"SACRIFICE_PERIOD_text-embedding-3-large_embeddings",
                                            "name":"embeddings of the sacrifice periods",
                                            "embedder": "text-embedding-3-large_embeddings",
                                            "embedding_size":3072}]}

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

input_config_data = {"feature_sets":[feature_set_molformer,feature_set_moldes,feature_set_best_model,feature_set_only_compound_name],
                        "target_column":"target",
                        "save_folder":"C:/Users/Tommaso/Documents/Dev/Cynde/cynde_mount/",
                        "remote_folder":"/cynde_mount"}

classifiers_config = ClassifierConfig(classifiers=[RandomForestClassifierConfig(n_estimators=1000, max_depth=15)])

input_config = InputConfig.model_validate(input_config_data,context={"df":df})

inner_groups = ["DOSE_LEVEL","SACRIFICE_PERIOD","target"]
outer_groups = ["COMPOUND_NAME"]

cv_config = CVConfig(inner= StratifiedConfig(groups=inner_groups,k=5),
                     inner_replicas=5,
                     outer = PurgedConfig(groups=outer_groups,k=5),
                        outer_replicas=10)

task = PredictConfig(input_config=input_config, cv_config=cv_config, classifiers_config=classifiers_config)






df_filtered = df.select(get_unique_columns(task))

preprocess_inputs(df_filtered,task.input_config)
# results = train_nested_cv(df,task)

results = train_nested_cv_distributed(df_filtered,task)
save_results(results,"tgate_with_moldes")
merged_results = merge_results(results)
merged_metrics = merge_metrics(results)
aggr_metrics = aggr_metrics_by_inputs(merged_metrics)
print(aggr_metrics)
