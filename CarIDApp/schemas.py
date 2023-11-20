from fastapi import UploadFile
from pydantic import BaseModel

class PredictionResult(BaseModel):
    class_name: str
    confidence: float
    execution_time: float

class ImageData(BaseModel):
    file: UploadFile
