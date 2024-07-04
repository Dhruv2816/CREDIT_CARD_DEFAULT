import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            model = RandomForestClassifier(
                bootstrap=True,
                ccp_alpha=0.0,
                class_weight=None,
                criterion='gini',
                max_depth=None,
                max_features='sqrt',
                max_leaf_nodes=None,
                max_samples=None,
                min_impurity_decrease=0.0,
                min_samples_leaf=1,
                min_samples_split=2,
                min_weight_fraction_leaf=0.0,
                n_estimators=100,
                n_jobs=None,
                oob_score=False,
                random_state=None,
                verbose=0,
                warm_start=False
            )

            model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info(f"Model training completed")

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            logging.info(f"Model Evaluation: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1-Score={f1}")

            return accuracy, precision, recall, f1
        except Exception as e:
            raise CustomException(e, sys)

















