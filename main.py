import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

def main():
    try:
        logging.info("Starting the pipeline.")

        # Data Ingestion
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()
        
        logging.info("Data ingestion completed.")

        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
        
        logging.info("Data transformation completed.")

        # Model Training
        model_trainer = ModelTrainer()
        accuracy, precision, recall, f1 = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        logging.info(f"Model training completed. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise e

if __name__ == "__main__":
    main()
