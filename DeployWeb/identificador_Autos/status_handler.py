import datetime
import psutil
from predictor import CarBrandPredictor

predictor = CarBrandPredictor()

def get_model_status(predictor):
    status_data = {
        "model_architecture": predictor.get_model_architecture(),
        "model_hyperparameters": predictor.get_model_hyperparameters(),
        "model_training_date": predictor.get_model_training_date(),
        "last_prediction": predictor.get_last_prediction(),
        "cpu_usage": predictor.get_cpu_usage(),  
        "memory_usage": predictor.get_memory_usage() 
    }
    return status_data


def get_cpu_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    return f"{cpu_usage}%"

def get_memory_usage():
    memory = psutil.virtual_memory()
    return f"Used: {memory.used / (1024 ** 3):.2f} GB, Total: {memory.total / (1024 ** 3):.2f} GB"