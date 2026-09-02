import os
import sys
import pandas as pd
import numpy as np 

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:

    train_data_path: str = os.path.join(
        "artifacts",
        "train.csv"
    )

    test_data_path: str = os.path.join(
        "artifacts",
        "test.csv"
    )

    raw_data_path: str = os.path.join(
        "artifacts",
        "data.csv"
    )


class DataIngestion:

    def __init__(self):

        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info(
            "Enter the data ingestion method"
        )

        try:

            # Read datasets
            client_df = pd.read_csv(
                "notebook\Client.csv"
            )

            record_df = pd.read_csv(
                "notebook\Record.csv"
            )

            logging.info(
                "Client and Record datasets loaded successfully"
            )

            # Merge datasets
            df = pd.merge(
                client_df,
                record_df,
                on="Customer_ID",
                how="inner"
            )

            logging.info(
                "Client and Record datasets merged successfully"
            )

            # Create artifacts directory
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )

            # Save raw data
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False
            )

            logging.info(
                "Train test split initiated"
            )

            # Train-test split
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["churn"]
            )

            # Save train data
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False
            )

            # Save test data
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False
            )

            logging.info(
                "Ingestion of the data is completed"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:

            raise CustomException(e, sys)


if __name__ == "__main__":

    obj = DataIngestion()

    train_data, test_data = (
        obj.initiate_data_ingestion()
    )

    # Data Transformation
    data_transformation = DataTransformation()

    (
        train_selected,
        test_selected,
        y_train,
        y_test
    ) = data_transformation.initiate_data_transformation(
        train_data,
        test_data
    )

    # Combine X and y for ModelTrainer
    train_arr = np.column_stack(
        (train_selected, y_train)
    )

    test_arr = np.column_stack(
        (test_selected, y_test)
    )

    # Model Training
    model_trainer = ModelTrainer()

    model_score = model_trainer.initiate_model_trainer(
        train_arr,
        test_arr
    )

    logging.info(
        f"Model training completed. F1 Score: {model_score}"
    )