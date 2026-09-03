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

from src.exception import CustomException
from src.logger import logging

from src.utils import (
    save_object,
    evaluate_models,
    tune_random_forest
)


@dataclass
class ModelTrainerConfig:

    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Split training and test input data"
            )

            # -----------------------------
            # Split X and y
            # -----------------------------

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]


            # -----------------------------
            # Models
            # -----------------------------

            models = {

                "Random Forest":
                    RandomForestClassifier(
                        random_state=42
                    ),

                "Decision Tree":
                    DecisionTreeClassifier(
                        random_state=42
                    ),

                "Gradient Boosting":
                    GradientBoostingClassifier(
                        random_state=42
                    ),

                "AdaBoost":
                    AdaBoostClassifier(
                        random_state=42
                    ),

                "XGBoost":
                    XGBClassifier(
                        random_state=42,
                        eval_metric="logloss"
                    )
            }


            # -----------------------------
            # Train all models normally
            # -----------------------------

            model_report, trained_models = evaluate_models(

                X_train=X_train,

                y_train=y_train,

                X_test=X_test,

                y_test=y_test,

                models=models
            )


            # -----------------------------
            # Find best model
            # -----------------------------

            best_model_score = max(
                model_report.values()
            )

            best_model_name = max(
                model_report,
                key=model_report.get
            )


            print("\n")
            print("*" * 60)
            print(
                f"Best Model Before Tuning: "
                f"{best_model_name}"
            )
            print(
                f"Best F1 Score Before Tuning: "
                f"{best_model_score:.4f}"
            )
            print("*" * 60)


            logging.info(
                f"Best model before tuning: "
                f"{best_model_name}"
            )

            logging.info(
                f"Best F1 score before tuning: "
                f"{best_model_score}"
            )


            # -----------------------------
            # Tune Random Forest ONLY
            # -----------------------------

            if best_model_name == "Random Forest":

                tuned_model, tuned_score = tune_random_forest(

                    X_train=X_train,

                    y_train=y_train,

                    X_test=X_test,

                    y_test=y_test
                )

                # Use tuned model
                best_model = tuned_model

                best_model_score = tuned_score


            else:

                # Use normal best model
                best_model = trained_models[
                    best_model_name
                ]


            # -----------------------------
            # Final Best Model
            # -----------------------------

            print("\n")
            print("=" * 60)
            print("FINAL MODEL")
            print("=" * 60)

            if best_model_name == "Random Forest":

                print(
                    "Model: Tuned Random Forest"
                )

            else:

                print(
                    f"Model: {best_model_name}"
                )

            print(
                f"Final F1 Score: "
                f"{best_model_score:.4f}"
            )

            print("=" * 60)


            # -----------------------------
            # Check score
            # -----------------------------

            if best_model_score < 0.6:

                raise CustomException(
                    "No best model found"
                )


            # -----------------------------
            # Save best model
            # -----------------------------

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

            print(
                "\nBest model saved successfully."
            )


            return best_model_score


        except Exception as e:

            raise CustomException(e, sys)