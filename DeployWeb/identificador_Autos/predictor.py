import time
import numpy as np
import psutil
from ultralytics import YOLO
from PIL import Image
import datetime


class CarBrandPredictor:
    MODEL_PATH = "C:/Users/gabon/Dev/PrimerParcialIA/PrimerParcial-TopicsIA/Training/runs/classify/train21/weights/best.pt"

    def __init__(self):
        print("Creando predictor...")
        self.predictions = []
        self.file_names = []
        self.model = YOLO(self.MODEL_PATH)
        self.model_architecture = "YOLO"
        self.model_hyperparameters = {"learning_rate": 0.001, "batch_size": 150, "num_epochs": 25, "image_size": 512}
        self.training_date = datetime.datetime.now()
        self.last_prediction = None

    def get_model_architecture(self):
        return self.model_architecture

    def get_model_hyperparameters(self):
        return self.model_hyperparameters

    def get_model_training_date(self):
        return self.training_date

    def get_last_prediction(self):
        return self.last_prediction
    
    def add_prediction(self, prediction):
        self.predictions.append(prediction)

    def predict_image(self, image_array: np.ndarray):
        file_name = f"example_file_{len(self.predictions) + 1}.jpg" 
        start_time = time.time()
        results = self.model(image_array)[0]
        end_time = time.time()

        prediction = {
            "file_name": file_name,
            "class": results.names[results.probs.top1],
            "confidence": results.probs.data[results.probs.top1].item(),
            "prediction_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time": end_time - start_time,
            "image_size": f"{self.model_hyperparameters['image_size']}x{self.model_hyperparameters['image_size']}",
            "model": self.model_architecture,
        }
        self.last_prediction = prediction
        self.predictions.append(prediction)
        self.file_names.append(file_name)  
        return prediction

    
    def get_all_file_names(self):
        return self.file_names
    
    def get_all_predictions(self):
        return self.predictions

    def get_cpu_usage(self):
        return psutil.cpu_percent()

    def get_memory_usage(self):
        return psutil.virtual_memory().percent
    

    def get_report_data(self):
        if self.last_prediction is None:
            return None

        report_data = {
            "file_name": self.last_prediction.get("file_name", ""),
            "image_size": f"{self.model_hyperparameters['image_size']}x{self.model_hyperparameters['image_size']}",
            "prediction": self.last_prediction.get("class", ""),
            "confidence": self.last_prediction.get("confidence", 0.0),
            "prediction_time": self.last_prediction.get("prediction_time", ""),
            "execution_time": self.last_prediction.get("execution_time", 0.0),
            "model": self.model_architecture,
        }

        return report_data