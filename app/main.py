import uvicorn
from fastapi import FastAPI
from app.schemas import PredictRequest, PredictResponse
from app.inference import predict
from app.model_loader import load_model
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

logger.info("Starting the Credit Card Default Prediction API")  

model = load_model()    
logger.info("Model loaded successfully")

@app.get("/")
def root() -> dict:
    logger.info("Root endpoint called")
    return {"message": "Welcome to the Credit Card Default Prediction API"}


@app.post("/predict")   
def predict_endpoint(request: PredictRequest)-> PredictResponse:
    logger.info(f"Predict endpoint called with request: {request}")
    response = predict(request, model)
    logger.info(f"Predict endpoint returned response: {response}")
    return response

if __name__ == "__main__":
    logger.info("Starting the Credit Card Default Prediction API")
    uvicorn.run(app, host="0.0.0.0", port=8000)