from cynde.functional.distributed_cv import train_nested_cv_from_np_modal, cv_stub, preprocess_np_modal
import cynde.functional as cf
import os
import polars as pl
from typing import List
import time

def load_minihermes_data(data_path: str = r"C:\Users\Tommaso\Documents\Dev\Cynde\cache\OpenHermes-2.5_embedded.parquet") -> pl.DataFrame:
    return pl.read_parquet(data_path)

df = load_minihermes_data()
print(df)


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




    