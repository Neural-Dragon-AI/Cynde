import os
import polars as pl

cynde_dir= r"/Users/tommasofurlanello/Documents/Dev/Cynde/"
cynde_dir = r"C:\Users\Tommaso\Documents\Dev\Cynde"
cache_dir = os.path.join(cynde_dir, "cache")
mount_dir = os.path.join(cynde_dir, "cynde_mount")
print(cache_dir)
data_name = r"tgca_nogen_simplified_smiles_malformer_embeddings_ada02_3large_3small_embeddings_grouped.parquet"


dataset_path = os.path.join(cache_dir, data_name)
print(dataset_path)

df = pl.read_parquet(dataset_path)
print(df.head(1))
print(df["COMPOUND_NAME"].value_counts())


print(df["target"].value_counts())