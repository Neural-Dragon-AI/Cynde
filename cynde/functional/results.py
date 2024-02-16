import polars as pl

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