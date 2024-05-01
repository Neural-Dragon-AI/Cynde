from enum import Enum
import polars as pl
from pydantic import BaseModel, ValidationInfo, model_validator,Field,ValidationInfo, field_validator

from enum import Enum
from typing import Optional, Union, Dict, Literal, Any, List, Tuple, Type, TypeVar, Generator


from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer, OneHotEncoder



class ScalerType(str, Enum):
    STANDARD_SCALER = "StandardScaler"
    MIN_MAX_SCALER = "MinMaxScaler"
    MAX_ABS_SCALER = "MaxAbsScaler"
    ROBUST_SCALER = "RobustScaler"
    POWER_TRANSFORMER = "PowerTransformer"
    QUANTILE_TRANSFORMER = "QuantileTransformer"
    NORMALIZER = "Normalizer"

class Feature(BaseModel):
    column_name: str
    name: str
    description: Optional[str] = None

    @field_validator("column_name")
    @classmethod
    def column_in_df(cls, v: str, info: ValidationInfo):
        column_name = v
        context = info.context
        if context:
            df = context.get("df",pl.DataFrame())
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        return v


class NumericalFeature(Feature):
    scaler_type: ScalerType = Field(ScalerType.STANDARD_SCALER, description="The type of scaler to apply to the numerical feature.")

    def get_scaler(self):
        scaler_map = {
            ScalerType.STANDARD_SCALER: StandardScaler(),
            ScalerType.MIN_MAX_SCALER: MinMaxScaler(),
            ScalerType.MAX_ABS_SCALER: MaxAbsScaler(),
            ScalerType.ROBUST_SCALER: RobustScaler(),
            ScalerType.POWER_TRANSFORMER: PowerTransformer(),
            ScalerType.QUANTILE_TRANSFORMER: QuantileTransformer(),
            ScalerType.NORMALIZER: Normalizer(),
        }
        return scaler_map[self.scaler_type]
    
    @field_validator("column_name")
    @classmethod
    def column_correct_type(cls, v: str, info: ValidationInfo):
        column_name = v
        context = info.context
        if context:
            df = context.get("df",pl.DataFrame())
            if df[column_name].dtype not in [pl.Boolean, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64, pl.Decimal]:
                current_dtype = df[column_name].dtype
                raise ValueError(f"Column '{column_name}'  with dtype {current_dtype} must be of a numeric type (Boolean, Integer, Unsigned Integer, Float, or Decimal) .")

        return v


class EmbeddingFeature(NumericalFeature):
    embedder: str = Field("text-embedding-3-small", description="The embedder model that generated the vector.")
    embedding_size:int = Field(1536, description="The size of the embedding vector.")

    @field_validator("column_name")
    @classmethod
    def column_correct_type(cls, v: str, info: ValidationInfo):
        column_name = v
        context = info.context
        if context:
            df = context.get("df",pl.DataFrame())
            if df[column_name].dtype not in [pl.List(pl.Float32), pl.List(pl.Float64)]:
                current_dtype = df[column_name].dtype
                raise ValueError(f"Column '{column_name}'  with dtype {current_dtype} must be of type pl.List(pl.Float32) or pl.List(pl.Float64).")
        return v

class CategoricalFeature(Feature):
    one_hot_encoding: bool = Field(True, description="Whether to apply one-hot encoding to the categorical feature.")

    @field_validator("column_name")
    @classmethod
    def column_correct_type(cls, v: str, info: ValidationInfo):
        column_name = v
        context = info.context
        if context:
            df = context.get("df",pl.DataFrame())
            if df[column_name].dtype not in [
                pl.Utf8,
                pl.Categorical,
                pl.Enum,
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            ]:
                current_dtype = df[column_name].dtype
                raise ValueError(
                    f"Column '{column_name}' with dtype {current_dtype}  must be of type pl.Utf8, pl.Categorical, pl.Enum, or an integer type."
                )
        return v

class FeatureSet(BaseModel):
    numerical: List[NumericalFeature] = []
    embeddings: List[EmbeddingFeature] = []
    categorical: List[CategoricalFeature] = []


    def all_features(self):
        return self.numerical + self.embeddings + self.categorical
    def column_names(self):
        return [feature.column_name for feature in self.all_features()]
    def joined_names(self):
        return "_".join(sorted(self.column_names()))

class InputConfig(BaseModel):
    feature_sets: List[FeatureSet]
    target_column: str = Field("target", description="The target column to predict.")
    save_folder: Optional[str] = None
    remote_folder: Optional[str] = None



class ClassifierName(str, Enum):
    LOGISTIC_REGRESSION = "LogisticRegression"
    RANDOM_FOREST = "RandomForestClassifier"
    HIST_GRADIENT_BOOSTING = "HistGradientBoostingClassifier"

class BaseClassifierConfig(BaseModel):
    classifier_name: ClassifierName
    

class LogisticRegressionConfig(BaseClassifierConfig):
    classifier_name: Literal[ClassifierName.LOGISTIC_REGRESSION] = Field(ClassifierName.LOGISTIC_REGRESSION)
    n_jobs: int = Field(-1, description="Number of CPU cores to use.")
    penalty: str = Field("l2", description="Specify the norm of the penalty.")
    dual: bool = Field(False, description="Dual or primal formulation.")
    tol: float = Field(1e-4, description="Tolerance for stopping criteria.")
    C: float = Field(1.0, description="Inverse of regularization strength.")
    fit_intercept: bool = Field(True, description="Specifies if a constant should be added to the decision function.")
    intercept_scaling: float = Field(1, description="Scaling factor for the constant.")
    class_weight: Optional[Union[str, Dict[Any, float]]] = Field(None, description="Weights associated with classes.")
    random_state: Optional[int] = Field(None, description="Seed for random number generation.")
    solver: str = Field("lbfgs", description="Algorithm to use in the optimization problem.")
    max_iter: int = Field(100, description="Maximum number of iterations.")
    multi_class: str = Field("auto", description="Approach for handling multi-class targets.")
    verbose: int = Field(0, description="Verbosity level.")
    warm_start: bool = Field(False, description="Reuse the solution of the previous call to fit.")
    l1_ratio: Optional[float] = Field(None, description="Elastic-Net mixing parameter.")

class RandomForestClassifierConfig(BaseClassifierConfig):
    classifier_name: Literal[ClassifierName.RANDOM_FOREST] = Field(ClassifierName.RANDOM_FOREST)
    n_jobs: int = Field(-1, description="Number of CPU cores to use.")
    n_estimators: int = Field(100, description="The number of trees in the forest.")
    criterion: str = Field("gini", description="The function to measure the quality of a split.")
    max_depth: Optional[int] = Field(None, description="The maximum depth of the tree.")
    min_samples_split: Union[int, float] = Field(2, description="The minimum number of samples required to split an internal node.")
    min_samples_leaf: Union[int, float] = Field(1, description="The minimum number of samples required to be at a leaf node.")
    min_weight_fraction_leaf: float = Field(0.0, description="The minimum weighted fraction of the sum total of weights required to be at a leaf node.")
    max_features: Union[str, int, float] = Field("sqrt", description="The number of features to consider when looking for the best split.")
    max_leaf_nodes: Optional[int] = Field(None, description="Grow trees with max_leaf_nodes in best-first fashion.")
    min_impurity_decrease: float = Field(0.0, description="A node will be split if this split induces a decrease of the impurity greater than or equal to this value.")
    bootstrap: bool = Field(True, description="Whether bootstrap samples are used when building trees.")
    oob_score: bool = Field(False, description="Whether to use out-of-bag samples to estimate the generalization score.")
    
    random_state: Optional[int] = Field(None, description="Seed for random number generation.")
    verbose: int = Field(0, description="Verbosity level.")
    warm_start: bool = Field(False, description="Reuse the solution of the previous call to fit and add more estimators to the ensemble.")
    class_weight: Optional[Union[str, Dict[Any, float]]] = Field(None, description="Weights associated with classes.")
    ccp_alpha: float = Field(0.0, description="Complexity parameter used for Minimal Cost-Complexity Pruning.")
    max_samples: Optional[Union[int, float]] = Field(None, description="If bootstrap is True, the number of samples to draw from X to train each base estimator.")
    monotonic_cst: Optional[Dict[str, int]] = Field(None, description="Monotonic constraint to enforce on each feature.")

class HistGradientBoostingClassifierConfig(BaseClassifierConfig):
    classifier_name: Literal[ClassifierName.HIST_GRADIENT_BOOSTING] = Field(ClassifierName.HIST_GRADIENT_BOOSTING)
    loss: str = Field("log_loss", description="The loss function to use in the boosting process.")
    learning_rate: float = Field(0.1, description="The learning rate, also known as shrinkage.")
    max_iter: int = Field(100, description="The maximum number of iterations of the boosting process.")
    max_leaf_nodes: int = Field(31, description="The maximum number of leaves for each tree.")
    max_depth: Optional[int] = Field(None, description="The maximum depth of each tree.")
    min_samples_leaf: int = Field(20, description="The minimum number of samples per leaf.")
    l2_regularization: float = Field(0.0, description="The L2 regularization parameter.")
    max_features: Union[str, int, float] = Field(1.0, description="Proportion of randomly chosen features in each and every node split.")
    max_bins: int = Field(255, description="The maximum number of bins to use for non-missing values.")
    categorical_features: Optional[Union[str, List[int], List[bool]]] = Field("warn", description="Indicates the categorical features.")
    monotonic_cst: Optional[Dict[str, int]] = Field(None, description="Monotonic constraint to enforce on each feature.")
    interaction_cst: Optional[Union[str, List[Tuple[int, ...]]]] = Field(None, description="Specify interaction constraints, the sets of features which can interact with each other in child node splits.")
    warm_start: bool = Field(False, description="Reuse the solution of the previous call to fit and add more estimators to the ensemble.")
    early_stopping: Union[str, bool] = Field("auto", description="Whether to use early stopping to terminate training when validation score is not improving.")
    scoring: Optional[str] = Field("loss", description="Scoring parameter to use for early stopping.")
    validation_fraction: float = Field(0.1, description="Proportion of training data to set aside as validation data for early stopping.")
    n_iter_no_change: int = Field(10, description="Used to determine when to stop if validation score is not improving.")
    tol: float = Field(1e-7, description="The absolute tolerance to use when comparing scores.")
    verbose: int = Field(0, description="Verbosity level.")
    random_state: Optional[int] = Field(None, description="Seed for random number generation.")
    class_weight: Optional[Union[str, Dict[Any, float]]] = Field(None, description="Weights associated with classes.")

class ClassifierConfig(BaseModel):
    classifiers: List[Union[LogisticRegressionConfig, RandomForestClassifierConfig, HistGradientBoostingClassifierConfig]]


class FoldMode(str, Enum):
    COMBINATORIAL = "Combinatorial"
    MONTE_CARLO = "MonteCarlo"

class BaseFoldConfig(BaseModel):
    k: int = Field(5, description="Number of folds. Divides the data into k equal parts. last k can be smaller or larger than the rest depending on the // of the data by k." )
    n_test_folds: int = Field(1, description="Number of test folds to use for cross-validation. Must be strictly less than k. if the fold mode is montecarlo they are sampled first and then the rest are used for training. If the fold mode is combinatorial the all symmetric combinations n_test out of k are sampled.")
    fold_mode: FoldMode = Field(FoldMode.COMBINATORIAL, description="The mode to use for splitting the data into folds. Combinatorial splits the data into k equal parts, while Monte Carlo randomly samples the k equal parts without replacement.")
    shuffle: bool = Field(True, description="Whether to shuffle the data before splitting.")
    random_state: Optional[int] = Field(None, description="Seed for random number generation. In the case of montecarlo cross-validation at each replica the seed is increased by 1 mantaining replicability while ensuring that the samples are different.")
    montecarlo_replicas: int = Field(5, description="Number of random replicas to use for montecarlo cross-validation.")

class KFoldConfig(BaseFoldConfig):
    pass

class StratificationMode(str, Enum):
    PROPORTIONAL = "Proportional"
    UNIFORM_STRICT = "UniformStrict"
    UNIFORM_RELAXED = "UniformRelaxed"

class StratifiedConfig(BaseFoldConfig):
    groups: List[str] = Field([], description="The df column(s) to use for stratification. They will be used for a group-by operation to ensure that the stratification is done within each group.")
    strat_mode : StratificationMode = Field(StratificationMode.PROPORTIONAL, description="The mode to use for stratification. Proportional ensures that the stratification is done within each group mantaining the original proportion of each group in the splits, this is done by first grouping and then breaking each group inot k equal parts, this ensure all the samples in each group are in train and test with the same proprtion. Uniform instead ensures that each group has the same number of samples in each train and test fold, this is not compatible with the proportional mode.")
    group_size : Optional[int] = Field(None, description="The number of samples to use for each group in the stratificaiton it will only be used if the strat_mode is uniform or uniform relaxed. If uniform relaxed is used the group size will be used as a target size for each group but if a group has less samples than the target size it will be used as is. If uniform strict is used the group_size for all groups will be forced to the min(group_size, min_samples_in_group).")

class PurgedConfig(BaseFoldConfig):
    groups: List[str] = Field([], description="The df column(s) to use for purging. They will be used for a group-by operation to ensure that the purging at the whole group level. K is going to used to determine the fraction of groups to purge from train and restrict to test. When the mode is montecarlo the groups are sampled first and then the rest are used for training. If the fold mode is combinatorial the all symmetric combinations n_test out of k groups partitions are sampled")

class CVConfig(BaseModel):
    inner: Union[KFoldConfig,StratifiedConfig,PurgedConfig]
    inner_replicas: int = Field(1, description="Number of random replicas to use for inner cross-validation.")
    outer: Optional[Union[KFoldConfig,StratifiedConfig,PurgedConfig]] = None
    outer_replicas: int = Field(1, description="Number of random replicas to use for outer cross-validation.")

class PredictConfig(BaseModel):
    cv_config: CVConfig
    input_config: InputConfig
    classifiers_config: ClassifierConfig

class CVSummary(BaseModel):
    cv_config: Union[KFoldConfig,StratifiedConfig,PurgedConfig] = Field(description="The cross-validation configuration. Required for the summary.")
    train_indexes: List[List[int]] = Field(description="The indexes of the training samples for each fold or replica.")
    test_indexes: List[List[int]] = Field(description="The indexes of the testing samples for each fold or replica.")
    fold_numbers: Optional[List[int]] = Field(None, description="The fold number for each sample. Used when the fold mode is combinatorial.")
    replica_numbers : Optional[List[int]] = Field(None, description="The replica number for each sample. Used when the fold mode is montecarlo.")
    
    def yield_splits(self) -> Generator[Tuple[pl.Series, pl.Series], None, None]:
        for train_idx, test_idx in zip(self.train_indexes, self.test_indexes):
            train_series = pl.Series(train_idx)
            test_series = pl.Series(test_idx)
            yield train_series, test_series

class PipelineInput(BaseModel):
    train_idx:pl.DataFrame
    val_idx:pl.DataFrame
    test_idx:pl.DataFrame
    feature_index:int
    cls_config:BaseClassifierConfig

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

class PipelineResults(BaseModel):
    train_predictions:pl.DataFrame
    val_predictions:pl.DataFrame
    test_predictions:pl.DataFrame
    train_accuracy:float
    train_mcc:float
    val_accuracy:float
    val_mcc:float
    test_accuracy:float
    test_mcc:float

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"



