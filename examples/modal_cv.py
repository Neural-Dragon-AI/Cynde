from cynde.functional.distributed_cv import train_nested_cv_from_np_modal, cv_stub
import cynde.functional as cf
import os
import polars as pl

stub = cv_stub
@stub.local_entrypoint()
def main():
    # Get the directory above the current directory
    cynde_dir = os.path.dirname(os.getcwd())

    # Specify the cache directory as 'data' subdirectory within the current directory
    # cynde_dir = os.path.join(above_dir, "Cynde")
    cynde_dir= r"/Users/tommasofurlanello/Documents/Dev/Cynde/"
    cache_dir = os.path.join(cynde_dir, "cache")
    mount_dir = os.path.join(cynde_dir, "cynde_mount")
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





    models_dict = {"RandomForest": [{"n_estimators": 10, "max_depth": 5},{"n_estimators": 50, "max_depth": 10}]}
    inputs =[{"numerical":["conversations_text-embedding-3-small_embedding"]},
            {"numerical":["conversations_text-embedding-3-large_embedding"]},
            {"numerical":["conversations_text-embedding-3-small_embedding","conversations_text-embedding-3-large_embedding"]}]
    # models_dict = {"RandomForest": [{"n_estimators": 10, "max_depth": 5}]}
    # inputs = [inputs[0]]

    # Call the train_nested_cv_from_np function with the required arguments
    df = cf.check_add_cv_index(df)

    results,pred=train_nested_cv_from_np_modal(df = embedded_df,
                     cv_type=("stratified","stratified"),
                     mount_dir=mount_dir,
                     inputs=inputs,
                     models=models_dict,
                     group_outer=["target"],
                     k_outer = 2,
                     group_inner=["target"],
                     k_inner = 2,
                     r_outer=10,
                     r_inner=10,
                     skip_class=False)# 


    print("Training completed successfully!")
    summary = cf.results_summary(results)
    print(summary)
    cf.save_results(df=df,results_df=results,pred_df=pred,save_name="test",base_path=cynde_dir)




    