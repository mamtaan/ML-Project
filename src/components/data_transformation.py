import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path: str = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )

    selector_obj_file_path: str = os.path.join(
        "artifacts",
        "selector.pkl"
    )


class DataTransformation:

    def __init__(self):

        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self, X_train):

        try:

            # Numerical columns
            numerical_columns = X_train.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()

            # Categorical columns
            categorical_columns = X_train.select_dtypes(
                include=["object", "string", "category"]
            ).columns.tolist()

            logging.info(
                f"Numerical columns: {len(numerical_columns)}"
            )

            logging.info(
                f"Categorical columns: {len(categorical_columns)}"
            )

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                     (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=True
                         )
                     )
                  ]
                )

            # Column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    ),
                    (
                        "cat_pipeline",
                        cat_pipeline,
                        categorical_columns
                    )
                ]
            )

            logging.info(
                "Preprocessing object created successfully"
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:

            logging.info(
                "Data transformation initiated"
            )

            # Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(
                "Train and test data loaded successfully"
            )

            target_column = "churn"

            # Separate features and target
            X_train = train_df.drop(
                columns=[target_column]
            )

            y_train = train_df[target_column]

            X_test = test_df.drop(
                columns=[target_column]
            )

            y_test = test_df[target_column]

            # Remove Customer_ID
            X_train = X_train.drop(
                columns=["Customer_ID"]
            )

            X_test = X_test.drop(
                columns=["Customer_ID"]
            )

            logging.info(
                "Customer_ID removed from features"
            )

            # Feature Engineering
            for df in [X_train, X_test]:

                df["revenue_per_call"] = (
                    df["totrev"] /
                    df["totcalls"].replace(0, np.nan)
                )

                df["minutes_per_call"] = (
                    df["totmou"] /
                    df["totcalls"].replace(0, np.nan)
                )

                df["revenue_per_month"] = (
                    df["totrev"] /
                    df["months"].replace(0, np.nan)
                )

                df["mou_change_recent"] = (
                    df["avg3mou"] -
                    df["avgmou"]
                )

                df["rev_change_recent"] = (
                    df["avg3rev"] -
                    df["avgrev"]
                )

                # Replace infinity values
                df.replace(
                    [np.inf, -np.inf],
                    np.nan,
                    inplace=True
                )

            logging.info(
                "Feature engineering completed"
            )

            # Create preprocessing object
            preprocessing_obj = (
                self.get_data_transformation_object(
                    X_train
                )
            )

            # Fit ONLY on training data
            X_train_transformed = (
                preprocessing_obj.fit_transform(
                    X_train
                )
            )

            # Transform test data
            X_test_transformed = (
                preprocessing_obj.transform(
                    X_test
                )
            )

            logging.info(
                "Preprocessing completed successfully"
            )

            # Feature Selection
            selector = SelectKBest(
                score_func=f_classif,
                k=50
            )

            # Fit selector ONLY on training data
            X_train_selected = selector.fit_transform(
                X_train_transformed,
                y_train
            )

            # Transform test data
            X_test_selected = selector.transform(
                X_test_transformed
            )

            logging.info(
                "Feature selection completed successfully"
            )

            # Save preprocessor
            save_object(
                file_path=(
                    self.data_transformation_config
                    .preprocessor_obj_file_path
                ),
                obj=preprocessing_obj
            )

            # Save selector
            save_object(
                file_path=(
                    self.data_transformation_config
                    .selector_obj_file_path
                ),
                obj=selector
            )

            logging.info(
                "Preprocessor and selector saved successfully"
            )

            logging.info(
                f"Training transformed shape: "
                f"{X_train_selected.shape}"
            )

            logging.info(
                f"Testing transformed shape: "
                f"{X_test_selected.shape}"
            )

            # Return transformed data
            return (
                X_train_selected,
                X_test_selected,
                y_train,
                y_test
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    obj = DataTransformation()