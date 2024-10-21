from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

pd.options.mode.copy_on_write = True


DataSplit = namedtuple(
    "DataSplit", ["test_x", "test_target", "train_x", "train_target"]
)


class BaseModel(ABC):
    model_slug: str = "base"

    def __init__(self, enable_categorical: bool = False) -> None:
        self.enable_categorical = enable_categorical

    @abstractmethod
    def baseline_encode(self, data_split: DataSplit) -> DataSplit:
        pass

    @abstractmethod
    def fit(self, train_x: pd.DataFrame, train_target: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, test_x: pd.DataFrame) -> np.ndarray:
        pass

    def handle_missing(
        self,
        categorical_columns: List[str],
        test_x: pd.DataFrame,
        train_x: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for col in train_x.columns:
            self._handle_missing_both(
                test_x[col], train_x[col], is_categorical=col in categorical_columns
            )
        return test_x, train_x

    @staticmethod
    def _handle_missing_both(
        test_x: pd.DataFrame, train_x: pd.DataFrame, is_categorical=True
    ) -> None:
        if is_categorical:
            train_x.fillna("missing", inplace=True)
            test_x.fillna("missing", inplace=True)
        else:
            train_x.fillna(train_x.mean(), inplace=True)
            test_x.fillna(test_x.mean(), inplace=True)

    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        try:
            df.columns = df.columns.str.replace("[", "(", regex=False)
        except:
            pass
        try:
            df.columns = df.columns.str.replace("]", ")", regex=False)
        except:
            pass
        try:
            df.columns = df.columns.str.replace("<", "lesser", regex=False)
        except:
            pass
        try:
            df.columns = df.columns.str.replace(">", "greater", regex=False)
        except:
            pass
        return df


class XGBoostModel(BaseModel):
    model_slug: str = "xgboost"

    def __init__(self, enable_categorical: bool = False, is_regression=False) -> None:
        super().__init__(enable_categorical)
        self.model = None
        self.is_regression = is_regression

    def baseline_encode(self, data_split: DataSplit) -> DataSplit:
        test_x = XGBoostModel.convert_objects_to_category(data_split.test_x)
        train_x = XGBoostModel.convert_objects_to_category(data_split.train_x)
        train_x = self.clean_column_names(train_x)
        test_x = self.clean_column_names(test_x)
        return DataSplit(
            test_x, data_split.test_target, train_x, data_split.train_target
        )

    def fit(self, train_x: pd.DataFrame, train_target: pd.Series):
        if self.is_regression:
            self.model = xgb.XGBRegressor(enable_categorical=self.enable_categorical)
            self.model.fit(train_x, train_target)
        else:
            self.model = xgb.XGBModel(
                enable_categorical=self.enable_categorical,
                num_class=len(train_target.unique()),
            )
            self.model.fit(train_x, train_target.astype("category").cat.codes)

    def predict(self, test_x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(test_x)

    @staticmethod
    def convert_objects_to_category(df: pd.DataFrame) -> pd.DataFrame:
        # Identify all object columns
        # Convert each object column to category type
        try:
            object_cols = df.select_dtypes(include=["object"]).columns
            df[object_cols] = df[object_cols].astype("category")
        except:
            pass
        return df
