from typing import List
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    file: bytes  # Cambia el tipo de datos según lo que esperas recibir

class ReportData(BaseModel):
    file_name: str
    image_size: str
    prediction: str
    confidence: float
    prediction_time: str
    execution_time: float
    model: str

class PredictAndAnnotateResponse(BaseModel):
    annotated_image: bytes  # Cambia el tipo de datos según lo que esperas devolver
