from cynde.functional.distributed_cv import train_nested_cv_from_np_modal, cv_stub, preprocess_np_modal
import cynde.functional as cf
import os
import polars as pl
from typing import List
import time
from cynde.functional.predict.types import StratifiedConfig,Feature,FeatureSet,NumericalFeature, CategoricalFeature,EmbeddingFeature, InputConfig, ClassifierConfig, LogisticRegressionConfig, RandomForestClassifierConfig, HistGradientBoostingClassifierConfig, CVConfig
from cynde.functional.predict.preprocess import convert_utf8_to_enum, check_add_cv_index, preprocess_inputs
from cynde.functional.predict.cv import stratified_combinatorial
from cynde.functional.predict.classify import create_pipeline

from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


def load_minihermes_data(data_path: str = r"C:\Users\Tommaso\Documents\Dev\Cynde\cache\OpenHermes-2.5_embedded.parquet") -> pl.DataFrame:
    return pl.read_parquet(data_path)

df = load_minihermes_data()
df = convert_utf8_to_enum(df, threshold=0.2)
df = check_add_cv_index(df,strict=False)
print(df.columns)

feature_set_small_data = {"embeddings":[{"column_name":"conversations_text-embedding-3-small_embeddings",
                                         "name":"feature set for the smaller oai embeddings"}]}
feature_set_large_data = {"embeddings":[{"column_name":"conversations_text-embedding-3-large_embeddings",
                                         "name":"feature set for the larger oai embeddings"}]}

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
df_idx = df.select(pl.col("cv_index"),pl.col(groups))

models = []
for classifiers in classifiers_config.classifiers:
    for feature_set in input_config.feature_sets:
        pipeline = create_pipeline(df, feature_set, classifiers)
        models.append(pipeline)
print(models)


def evaluate_model(pipeline: Pipeline, X, y):
    predictions = pipeline.predict(X)
    accuracy = accuracy_score(y, predictions)
    return accuracy

for r_o in range(cv_config.outer_replicas):
    outer_cv = stratified_combinatorial(df_idx, cv_config.outer)
    for k_o, (dev_idx_o,test_idx_o) in enumerate(outer_cv.yield_splits()):
        df_test_idx_o = df_idx.filter(pl.col("cv_index").is_in(test_idx_o))
        df_dev_idx_o = df_idx.filter(pl.col("cv_index").is_in(dev_idx_o))
        for r_i in range(cv_config.inner_replicas):
            inner_cv = stratified_combinatorial(df_dev_idx_o, cv_config.inner)
            for k_i,(train_idx_i,val_idx_i) in enumerate(inner_cv.yield_splits()):
                df_val_idx_o_i = df_idx.filter(pl.col("cv_index").is_in(val_idx_i))
                df_train_idx_o_i = df_idx.filter(pl.col("cv_index").is_in(train_idx_i))
                n_train = df_train_idx_o_i.shape[0]
                n_val = df_val_idx_o_i.shape[0]
                n_test = df_test_idx_o.shape[0]
                print(f"For outer replica {r_o}, outer fold {k_o}, inner replica {r_i}, inner fold {k_i}: train {n_train}, val {n_val}, test {n_test} samples.")
                df_train = df.filter(pl.col("cv_index").is_in(train_idx_i))
                df_val = df.filter(pl.col("cv_index").is_in(val_idx_i))
                df_test = df.filter(pl.col("cv_index").is_in(test_idx_o))
                pipeline.fit(df_train,df_train["target"])
                print("Fitted pipeline")
                  # Evaluate the model on the validation and test data
                val_accuracy = evaluate_model(pipeline, df_val, df_val["target"])
                test_accuracy = evaluate_model(pipeline, df_test, df_test["target"])

                print("Validation accuracy:", val_accuracy)
                print("Test accuracy:", test_accuracy)
                
                    
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




    