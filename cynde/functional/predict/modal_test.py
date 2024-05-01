from cynde.functional.distributed_cv import train_nested_cv_from_np_modal, cv_stub, preprocess_np_modal
import cynde.functional as cf
import os
import polars as pl
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

## here the distributed part start



# def evaluate_model(pipeline: Pipeline, X, y):
#     predictions = pipeline.predict(X)
#     accuracy = accuracy_score(y, predictions)
#     return predictions,accuracy

# df_idx = df.select(pl.col("cv_index"),pl.col(groups))

# def train_test_val(df:pl.DataFrame,train_idx,val_idx,test_idx):
#     print("training idx",train_idx)
#     df_train = df.filter(pl.col("cv_index").is_in(train_idx["cv_index"]))
#     df_val = df.filter(pl.col("cv_index").is_in(val_idx["cv_index"]))
#     df_test = df.filter(pl.col("cv_index").is_in(test_idx["cv_index"]))
#     return df_train,df_val,df_test


# def predict_pipeline(input_config:InputConfig,train_idx:pl.DataFrame,val_idx:pl.DataFrame,test_idx:pl.DataFrame,feature_index:int,classifier:BaseClassifierConfig):
#     feature_set = input_config.feature_sets[feature_index]
#     df_fold = load_preprocessed_features(input_config,feature_index)
#     print(df_fold)
#     df_train,df_val,df_test = train_test_val(df_fold,train_idx,val_idx,test_idx)
#     print(df_train)
#     pipeline = create_pipeline(df_train, feature_set, classifier)
#     print(pipeline)
#     pipeline.fit(df_train,df_train["target"])


#     val_predictions,val_accuracy = evaluate_model(pipeline, df_val, df_val["target"])
#     test_predictions,test_accuracy = evaluate_model(pipeline, df_test, df_test["target"])
#     return val_predictions,test_predictions,val_accuracy,test_accuracy

# #Outer Replicas
# def generate_nested_cv(df_idx:pl.DataFrame, cv_config:CVConfig, input_config:InputConfig, classifiers_config:ClassifierConfig) -> Generator[Tuple[pl.DataFrame,pl.DataFrame,pl.DataFrame,int,BaseClassifierConfig],None,None]:
#     for r_o in range(cv_config.outer_replicas):
#         outer_cv = stratified_combinatorial(df_idx, cv_config.outer)
#         #Outer Folds -- this is an instance of an outer cross-validation fold
#         for k_o, (dev_idx_o,test_idx_o) in enumerate(outer_cv.yield_splits()):
#             df_test_idx_o = df_idx.filter(pl.col("cv_index").is_in(test_idx_o))
#             df_dev_idx_o = df_idx.filter(pl.col("cv_index").is_in(dev_idx_o))
#             #Inner Replicas
#             for r_i in range(cv_config.inner_replicas):
#                 inner_cv = stratified_combinatorial(df_dev_idx_o, cv_config.inner)
#                 #Inner Folds -- this is an instance of an inner cross-validation fold
#                 for k_i,(train_idx_i,val_idx_i) in enumerate(inner_cv.yield_splits()):
#                     df_val_idx_o_i = df_idx.filter(pl.col("cv_index").is_in(val_idx_i))
#                     df_train_idx_o_i = df_idx.filter(pl.col("cv_index").is_in(train_idx_i))
#                     n_train = df_train_idx_o_i.shape[0]
#                     n_val = df_val_idx_o_i.shape[0]
#                     n_test = df_test_idx_o.shape[0]
#                     print(f"For outer replica {r_o}, outer fold {k_o}, inner replica {r_i}, inner fold {k_i}: train {n_train}, val {n_val}, test {n_test} samples.")
#                     #Feature types loop
#                     for feature_index,feature_set in enumerate(input_config.feature_sets):
#                         for classifier in classifiers_config.classifiers:
#                             yield df_train_idx_o_i, df_val_idx_o_i, df_test_idx_o, feature_index,classifier


# nested_cv = generate_nested_cv(df_idx, cv_config, input_config, classifiers_config)
# for df_train_idx_o_i, df_val_idx_o_i, df_test_idx_o, feature_index,classifier in nested_cv:
#     print(f"Training classifier {classifier.classifier_name} with feature set {input_config.feature_sets[feature_index].joined_names()}")
#     start_time = time.time()
#     val_accuracy,test_accuracy = predict_pipeline(input_config,df_train_idx_o_i,df_val_idx_o_i,df_test_idx_o,feature_index,classifier)
#     print("Fitted pipeline in", time.time() - start_time, "seconds.")
#     print("Validation accuracy:", val_accuracy)
#     print("Test accuracy:", test_accuracy)

                
                    
# preprocess_np_modal(df = df_f_ne, mount_dir = mount_dir, inputs = inputs, target_column = target)
# stub = cv_stub
# @stub.local_entrypoint()
# def main():
    
#     results,pred=train_nested_cv_from_np_modal(df = df_f_ne,
#                      cv_type=("resample","stratified"),
#                      mount_dir=mount_dir,
#                      inputs=inputs,
#                      models=models_dict,
#                      group_outer=[target],
#                      k_outer = 5,
#                      group_inner=[target],
#                      k_inner = 5,
#                      r_outer=1,
#                      r_inner=1,
#                      skip_class=False,
#                      target_column = target)# 


#     print("Training completed successfully!")
#     summary = cf.results_summary(results)
#     print(summary)
#     cf.save_results(df=df_f_ne,results_df=results,pred_df=pred,save_name="test",base_path=cynde_dir)




    