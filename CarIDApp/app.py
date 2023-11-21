import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException, status, Depends
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from schemas import ImageData, PredictionResult
from predictor import CarBrandPredictor
from status_handler import get_model_status


app = FastAPI(title="Car Brand Classification API")

predictor = CarBrandPredictor()

def get_predictor():
    return predictor

## /get primer endpoint de status
@app.get("/status")
def get_status():
    return get_model_status(predictor)


def predict_and_annotate(predictor: CarBrandPredictor, file: UploadFile):
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Not an image"
        )

    img_obj = Image.open(img_stream)
    img_array = np.array(img_obj)

    results = predictor.predict_image(img_array)

    annotated_image = annotate_image(img_array, results, results["execution_time"])

    annotated_image_bytes = image_to_bytes(annotated_image)

    return annotated_image_bytes

def annotate_image(img_array: np.ndarray, results: dict, execution_time: float):
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    confidence_threshold = 0.50
    if results['confidence'] < confidence_threshold:
        class_label = "Marca Desconocida"
    else:
        class_label = results['class']

    text_size = max(1, int(min(img_array.shape[0], img_array.shape[1]) / 500)) 

    bar_height = 100  
    img_with_bar = np.zeros((img_array.shape[0] + bar_height, img_array.shape[1], img_array.shape[2]), dtype=np.uint8)
    img_with_bar[:img_array.shape[0], :, :] = img_rgb

    text_position = (10, img_array.shape[0] + 30)

    lines = [
        f"{class_label}",
        f"Confi: {results['confidence']:.2%}",
        f"Time: {execution_time:.4f} seconds",
    ]

    for i, line in enumerate(lines):
        y = text_position[1] + i * 30 
        annotated_img = cv2.putText(
            img_with_bar,
            line,
            (text_position[0], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            (255, 0, 0),
            2,
        )

    return annotated_img


def image_to_bytes(img_array: np.ndarray):
    img_pil = Image.fromarray(img_array)
    img_stream = io.BytesIO()
    img_pil.save(img_stream, format="JPEG")
    img_stream.seek(0)
    return img_stream.read()

### /post segundo endpoint para predecir y anotar en la imagen predicha

@app.post("/predict_and_annotate", response_model=PredictionResult)
def predict_and_annotate_route(
    image_data: ImageData = Depends(),
    predictor: CarBrandPredictor = Depends(get_predictor)
):
    annotated_image_bytes = predict_and_annotate(predictor, image_data.file)
    return StreamingResponse(io.BytesIO(annotated_image_bytes), media_type="image/jpeg")


### /get tercer endpoint para generar reportes con informacion de als predicciones hechas

@app.get("/reports", response_class=Response, responses={200: {"content": {"text/csv": {}}}})
def generate_reports(predictor: CarBrandPredictor = Depends(get_predictor)):
    all_predictions = predictor.get_all_predictions()

    if not all_predictions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No predictions available to generate a report."
        )

    csv_content = "file_name,image_size,prediction,confidence,prediction_time,execution_time,model\n"
    for prediction in all_predictions:
        csv_content += f"{prediction['file_name']},{prediction['image_size']},{prediction['class']},{prediction['confidence']},{prediction['prediction_time']},{prediction['execution_time']},{prediction['model']}\n"

    response = Response(content=csv_content, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=reports.csv"})

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", reload=True)