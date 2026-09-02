import os
import sys

from dataclasses import dataclass

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from xgboost import XGBClassifier

from sklearn.metrics import f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:

    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = (
            ModelTrainerConfig()
        )

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Split training and test input data"
            )

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {

                "Random Forest":
                    RandomForestClassifier(),

                "Decision Tree":
                    DecisionTreeClassifier(),

                "Gradient Boosting Classifier":
                    GradientBoostingClassifier(),

                "Ada Boost Classifier":
                    AdaBoostClassifier(),

                "XGBoost Classifier":
                    XGBClassifier()
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            # Get best model score
            best_model_score = max(
                model_report.values()
            )

            # Get best model name
            best_model_name = max(
                model_report,
                key=model_report.get
            )

            # Get best model
            best_model = models[best_model_name]

            logging.info(
                f"Best model: {best_model_name}"
            )

            logging.info(
                f"Best model F1 score: {best_model_score}"
            )

            if best_model_score < 0.6:

                raise CustomException(
                    "No best model found"
                )
            print(f"Best Model: {best_model_name}")
            print(f"Best F1 Score: {best_model_score:.4f}")
            
            # Save best model
            save_object(
                file_path=(
                    self.model_trainer_config
                    .trained_model_file_path
                ),
                obj=best_model
            )

            logging.info(
                "Best model saved successfully"
            )

            return best_model_score

        except Exception as e:

            raise CustomException(e, sys)


    def evaluate_models(
            X_train,
            y_train,
            X_test,
            y_test,
            models
        ):

        try:

            report = {}

            for model_name, model in models.items():

                logging.info(
                    f"Training {model_name}"
                )

                # Train model
                model.fit(
                    X_train,
                    y_train
                )

                # Predictions
                y_test_pred = model.predict(
                    X_test
                )

                # F1 score
                test_model_score = f1_score(
                    y_test,
                    y_test_pred
                )

                report[model_name] = test_model_score

                logging.info(
                    f"{model_name} F1 Score: "
                    f"{test_model_score}"
                )

            return report

        except Exception as e:
         raise CustomException(e, sys)