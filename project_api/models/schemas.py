from enum import Enum
from fastapi import FastAPI, Body
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel, ValidationError


class ModelType(str, Enum):
    sklearn = "sklearn"
    xgb = "xgb"
    lgb = "lgb"


class PredictionResponse(BaseModel):
    status_code: int
    predictions: list = []
    errorMessage : str = ''
