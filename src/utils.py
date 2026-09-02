import os
import sys
import dill

from sklearn.metrics import f1_score
from src.exception import CustomException


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

        for model_name, model in models.items():

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

        return report

    except Exception as e:

        raise CustomException(e, sys)