# app/models.py
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import numpy as np

class PredictionRequest(BaseModel):
    """
    Схема запроса на предсказание.
    Ожидает массив признаков в виде списка чисел.
    """
    features: List[float] = Field(
        ..., 
        description="Список признаков для предсказания",
        examples=[[25.0, 10.0, 5.0, 2.0]]
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": [25.0, 10.0, 5.0, 2.0]
            }
        }
    )

class PredictionResponse(BaseModel):
    """
    Схема ответа с предсказанием.
    """
    prediction: float = Field(..., description="Предсказанное значение")
    model_used: str = Field(..., description="Имя использованной модели")
    features_count: int = Field(..., description="Количество переданных признаков")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 110.5,
                "model_used": "xgboost_model",
                "features_count": 4
            }
        }
    )

class HealthResponse(BaseModel):
    """
    Схема ответа для проверки здоровья сервиса.
    """
    status: str = Field(..., description="Статус сервиса")
    model_loaded: bool = Field(..., description="Загружена ли модель")
    api_version: str = Field(..., description="Версия API")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "api_version": "1.0.0"
            }
        }
    )

class ErrorResponse(BaseModel):
    """
    Схема ошибки.
    """
    detail: str = Field(..., description="Описание ошибки")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Модель не загружена"
            }
        }
    )