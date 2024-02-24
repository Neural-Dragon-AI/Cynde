from datasets import load_dataset
import polars as pl
import os
import cynde.functional as cf

# Get the directory above the current directory
above_dir = os.path.dirname(os.getcwd())

# Specify the cache directory as 'data' subdirectory within the current directory
cynde_dir = os.path.join(above_dir, "Cynde")

cache_dir = os.path.join(cynde_dir, "cache")
print(cache_dir)

dataset_name = "OpenHermes-2.5"
dataset_path = os.path.join(cache_dir, dataset_name)


df = pl.read_parquet(dataset_path+"_filtered.parquet")
print(df.head(1))
print(df["source"].value_counts())

df = df.with_columns(pl.col("conversations").list.eval(pl.element().struct.json_encode()).list.join("\n"),(pl.col("source")=="caseus_custom").alias("target"))

#check if the embedded df already exist at dataset_path+"_embedded.parquet"
if not os.path.exists(dataset_path+"_embedded_small_large.parquet"):
    embedded_df = cf.embed_columns(df, ["conversations"],models=["text-embedding-3-small","text-embedding-3-large"])
    embedded_df.write_parquet(dataset_path+"_embedded_small_large.parquet")
else:
    embedded_df = pl.read_parquet(dataset_path+"_embedded_small_large.parquet")

print(df["target"].value_counts())

print(cf.vanilla_kfold(embedded_df,group=None,k=5))


models_dict = {"RandomForest": [{"n_estimators": 10, "max_depth": 5},{"n_estimators": 50, "max_depth": 10}]}
inputs =[{"numerical":["conversations_text-embedding-3-small_embeddings"]},
         {"numerical":["conversations_text-embedding-3-large_embeddings"]},
         {"numerical":["conversations_text-embedding-3-small_embeddings","conversations_text-embedding-3-large_embeddings"]}]
inputs = [inputs[0]]

results,pred=cf.train_nested_cv(df = embedded_df,
                     cv_type=("stratified","stratified"),
                     inputs=inputs,
                     models=models_dict,
                     group_outer=["target"],
                     k_outer = 2,
                     group_inner=["target"],
                     k_inner = 2,
                     r_outer=1,
                     r_inner=1,
                     save_name="test")


summary = cf.results_summary(results,by_test_fold=True)
print(summary)

for res in summary.sort(by="mcc_val",descending=True).rows(named=True):
    print(res["input_features_name"],res["classifier_hp"],res["mcc_val"])