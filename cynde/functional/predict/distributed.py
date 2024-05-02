import modal
from modal import Image
from typing import Tuple
from cynde.functional.predict.classify import predict_pipeline
from cynde.functional.predict.types import PipelineInput,PipelineResults,PredictConfig
from cynde.functional.predict.preprocess import load_preprocessed_features,check_add_cv_index
from cynde.functional.predict.cv import train_test_val,generate_nested_cv
from cynde.functional.predict.classify import create_pipeline ,evaluate_model



stub = modal.Stub("distributed_cv")
    
datascience_image = (
    Image.debian_slim(python_version="3.12.1")
    .apt_install("git")
    .pip_install("polars","scikit-learn","openai","tiktoken", force_build=True)
    
    .run_commands("git clone https://github.com/Neural-Dragon-AI/Cynde/")
    .env({"CYNDE_DIR": "/opt/cynde"})
    .run_commands("cd Cynde && pip install -r requirements.txt && pip install .")
)
with datascience_image.imports():
    import polars as pl
    import sklearn as sk
    import cynde as cy


#define the distributed classification method
@stub.function(image=datascience_image, mounts=[modal.Mount.from_local_dir(r"C:\Users\Tommaso\Documents\Dev\Cynde\cynde_mount", remote_path="/root/cynde_mount")])
def predict_pipeline_distributed(pipeline_input:PipelineInput) -> Tuple[pl.DataFrame,pl.DataFrame,float,float]:
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
                           test_mcc=test_mcc)

def train_nested_cv_distributed(df:pl.DataFrame,task_config:PredictConfig) -> pl.DataFrame:
    """ Deploy a CV training pipeline to Modal, it requires a df with cv_index column and the features set to have already pre-processed and cached 
    1) Validate the input_config and check if the preprocessed features are present locally 
    2) create a generator that yields the modal path to the features and targets frames as well as the scikit pipeline object 
    3) execute through a modal starmap a script that fit end eval each pipeline on each feature set and return the results
    4) collect and aggregate the results locally and save and return the results
    """
    #validate the inputs and check if the preprocessed features are present locally
    df = check_add_cv_index(df,strict=True)
    
    
    #extract the subset of columns necessary for constructing the cross validation folds 
    unique_groups = list(set(task_config.cv_config.inner.groups + task_config.cv_config.outer.groups))
    df_idx = df.select(pl.col("cv_index"),pl.col(unique_groups))

    nested_cv = generate_nested_cv(df_idx,task_config)
    all_results = []
    for result in predict_pipeline_distributed.map(list(nested_cv)):
        all_results.append(result)
    print("Finished!! " ,len(all_results))