from modal import Volume
import modal
from modal import Image
from typing import Tuple
from cynde.functional.train.types import PipelineInput,PipelineResults,InputConfig
from cynde.functional.train.preprocess import load_preprocessed_features
from cynde.functional.train.cv import train_test_val
from cynde.functional.train.train_local import create_pipeline ,evaluate_model
import os

from modal import Volume
vol = Volume.from_name("cynde_cv", create_if_missing=True)

app = modal.App("distributed_cv")
    
datascience_image = (
    Image.debian_slim(python_version="3.12.1")
    .apt_install("git")
    .pip_install("polars","scikit-learn","openai","tiktoken")#, force_build=True)
    
    .run_commands("git clone https://github.com/Neural-Dragon-AI/Cynde/")
    .env({"CYNDE_DIR": "/opt/cynde"})
    .run_commands("cd Cynde && pip install -r requirements.txt && pip install .")
)
with datascience_image.imports():
    import polars as pl
    import sklearn as sk
    import cynde as cy


@app.function(image=datascience_image,volumes={"/cynde_mount": vol})
def preprocess_inputs_distributed(df: pl.DataFrame, input_config: InputConfig):
    """ Saves .parquet for each feature set in input_config """
    save_folder = input_config.remote_folder
    os.makedirs(save_folder,exist_ok=True)
    for feature_set in input_config.feature_sets:
        column_names = feature_set.column_names()
        feature_set_df = df.select(pl.col("cv_index"),pl.col("target"),pl.col(column_names))
        print(f"selected columns: {feature_set_df.columns}")
        save_name = feature_set.joined_names()
        save_path = os.path.join(save_folder, f"{save_name}.parquet")
        feature_set_df.write_parquet(save_path)
        print("saved the parquet file at ", save_path)
    vol.commit()
    print("committed the volume")


#define the distributed classification method
@app.function(image=datascience_image,volumes={"/cynde_mount": vol})
def train_pipeline_distributed(pipeline_input:PipelineInput) -> Tuple[pl.DataFrame,pl.DataFrame,float,float]:
    input_config = pipeline_input.input_config
    feature_set = input_config.feature_sets[pipeline_input.feature_index]
    df_fold = load_preprocessed_features(input_config,pipeline_input.feature_index,remote=True)
    print(df_fold)
    df_train,df_val,df_test = train_test_val(df_fold,pipeline_input.train_idx,pipeline_input.val_idx,pipeline_input.test_idx)
    print(df_train)
    pipeline = create_pipeline(df_train, feature_set, pipeline_input.cls_config)
    print(pipeline)
    pipeline.fit(df_train,df_train["target"])
    train_predictions, train_accuracy, train_mcc = evaluate_model(pipeline, df_train, df_train["target"])
    val_predictions,val_accuracy, val_mcc = evaluate_model(pipeline, df_val, df_val["target"])
    test_predictions,test_accuracy,test_mcc = evaluate_model(pipeline, df_test, df_test["target"])
    return PipelineResults(train_predictions = train_predictions,
                           val_predictions=val_predictions,
                           test_predictions=test_predictions,
                           train_accuracy=train_accuracy,
                           train_mcc=train_mcc,
                           val_accuracy=val_accuracy,
                           val_mcc=val_mcc,
                           test_accuracy=test_accuracy,
                           test_mcc=test_mcc,
                           pipeline_input=pipeline_input)