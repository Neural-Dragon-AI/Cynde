import polars as pl
from cynde.functional.cv import get_fold_name_cv
from cynde.functional.classify import get_hp_classifier_name, get_pred_column_name, get_input_name
from typing import List, Dict, Union, Tuple, Any

def results_summary(results:pl.DataFrame,by_test_fold:bool=False) -> pl.DataFrame:
    groups = [ "classifier","classifier_hp","input_features_name"]
    if by_test_fold:
        groups += ["r_outer","r_inner"]
       
    summary = results.group_by(
   groups).agg(
    pl.col(["mcc_train","mcc_val","mcc_test"]).mean(),
     pl.col(["accuracy_train","accuracy_val","accuracy_test"]).mean(),
    pl.len().alias("n")).sort("mcc_val",descending=True)
    return summary



def get_predictions(joined_df:pl.DataFrame,
                    cv_type: Tuple[str, str],
                    inputs: List[Dict[str, Union[List[str], List[List[str]]]]],
                    models: Dict[str, List[Dict[str, Any]]],
                    group_outer: List[str],
                    k_outer: int,
                    group_inner: List[str],
                    k_inner: int,
                    r_outer: int = 1,
                    r_inner: int = 1,) -> pl.DataFrame:
    outs = []
    
    for r_o in range(r_outer):
        for k_o in range(k_outer):
            for r_i in range(r_inner):
                for k_i in range(k_inner):
                    fold_name = get_fold_name_cv(group_outer, cv_type, r_o,k_o,group_inner,r_i,k_i)
                    for input_feature in inputs:
                        input_name = get_input_name(input_feature)
                        for model, hp_list in models.items():
                            for hp in hp_list:
                                hp_name = get_hp_classifier_name(hp)
                                pred_col_name = get_pred_column_name(fold_name, input_name, model, hp_name)
                                if pred_col_name not in joined_df.columns:
                                    raise ValueError(f"Column {pred_col_name} not found in the joined_df")
                                outs.append((fold_name,pred_col_name))
    return outs

def get_all_predictions_by_inputs_model(joined_df:pl.DataFrame,
                    cv_type: Tuple[str, str],
                    inputs: List[Dict[str, Union[List[str], List[List[str]]]]],
                    models: Dict[str, List[Dict[str, Any]]],
                    group_outer: List[str],
                    k_outer: int,
                    group_inner: List[str],
                    k_inner: int,
                    r_outer: int = 1,
                    r_inner: int = 1,)  :
    
    for input_feature in inputs:
                        input_name = get_input_name(input_feature)
                        for model, hp_list in models.items():
                            for hp in hp_list:
                                hp_name = get_hp_classifier_name(hp)
                                pred_cols_by_model =[]
                                for r_o in range(r_outer):
                                    for k_o in range(k_outer):
                                        for r_i in range(r_inner):
                                            for k_i in range(k_inner):
                                                fold_name = get_fold_name_cv(group_outer, cv_type, r_o,k_o,group_inner,r_i,k_i)
                                                pred_col_name = get_pred_column_name(fold_name, input_name, model, hp_name)
                                                if pred_col_name not in joined_df.columns:
                                                    raise ValueError(f"Column {pred_col_name} not found in the joined_df")
                                                pred_cols_by_model.append((fold_name,pred_col_name))
                                yield input_name,model,hp_name,pred_cols_by_model