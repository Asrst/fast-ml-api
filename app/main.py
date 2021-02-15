import os, sys
import traceback
from enum import Enum
from fastapi import FastAPI, Body
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel, ValidationError
from models.schemas import ModelType, PredictionResponse
from models.predict import predict_async, predict_parallel_async
from models.loader import load_sklearn_joblib_model


app = FastAPI(title = 'fast-ml-api')
sklearn_model = load_sklearn_joblib_model('../models_store/NB.joblib')

@app.get("/")
async def health_check():
    return {"message": "Hello World"}


@app.post("/sklearn-model/predict", response_model=PredictionResponse)
async def make_prediction(inputs: List[List[float]] = Body(..., embed=True)):

    result_json = {'status_code': 500, 'predictions': [],
                    'errorMessage': ''}
    try:
        preds = await predict_async(model=sklearn_model, X=inputs)
        # preds = joblib_predict(sklearn_model, inputs)
        result_json['predictions'] = preds
        result_json['status_code'] = 200
    except Exception as e:
        print('Error Occurred in Sklearn Predict API:', e)
        traceback.print_exc()
        result_json['errorMessage'] = traceback.format_exc()
    return result_json




