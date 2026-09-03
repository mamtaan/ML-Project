import os
import sys
import dill

from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):

    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(
            dir_path,
            exist_ok=True
        )

        with open(
            file_path,
            "wb"
        ) as file_obj:

            dill.dump(
                obj,
                file_obj
            )

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
        trained_models = {}

        for model_name, model in models.items():

            logging.info(
                f"Training {model_name}"
            )

            # Train model
            model.fit(
                X_train,
                y_train
            )

            # Prediction
            y_test_pred = model.predict(
                X_test
            )

            # F1 Score
            test_model_score = f1_score(
                y_test,
                y_test_pred
            )

            report[model_name] = test_model_score
            trained_models[model_name] = model

            # Terminal output
            print("\n" + "=" * 60)
            print(f"Model: {model_name}")
            print(f"F1 Score: {test_model_score:.4f}")
            print("=" * 60)

            logging.info(
                f"{model_name} F1 Score: "
                f"{test_model_score:.4f}"
            )

        return report, trained_models

    except Exception as e:

        raise CustomException(e, sys)


def tune_random_forest(
    X_train,
    y_train,
    X_test,
    y_test
):

    try:

        logging.info(
            "Starting Random Forest hyperparameter tuning"
        )

        model = RandomForestClassifier(
            random_state=42
        )

        params = {

            "n_estimators": [
                100,
                200,
                300
            ],

            "max_depth": [
                10,
                20,
                30,
                None
            ],

            "min_samples_split": [
                2,
                5,
                10
            ],

            "min_samples_leaf": [
                1,
                2,
                4
            ],

            "max_features": [
                "sqrt",
                "log2",
                None
            ],

            "criterion": [
                "gini",
                "entropy"
            ]
        }

        random_search = RandomizedSearchCV(

            estimator=model,

            param_distributions=params,

            n_iter=5,

            cv=3,

            scoring="f1",

            random_state=42,

            n_jobs=-1
        )

        random_search.fit(
            X_train,
            y_train
        )

        # Best model
        best_model = random_search.best_estimator_

        # Prediction
        y_test_pred = best_model.predict(
            X_test
        )

        # F1 Score
        test_model_score = f1_score(
            y_test,
            y_test_pred
        )

        # Terminal output
        print("\n" + "*" * 60)
        print("TUNED RANDOM FOREST")
        print("*" * 60)

        print(
            f"F1 Score: "
            f"{test_model_score:.4f}"
        )

        print(
            f"Best Parameters: "
            f"{random_search.best_params_}"
        )

        print("*" * 60)

        logging.info(
            f"Tuned Random Forest F1 Score: "
            f"{test_model_score:.4f}"
        )

        logging.info(
            f"Tuned Random Forest Parameters: "
            f"{random_search.best_params_}"
        )

        return best_model, test_model_score

    except Exception as e:

        raise CustomException(e, sys)