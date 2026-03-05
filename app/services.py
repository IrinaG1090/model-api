# app/services.py
import joblib
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from app.config import MODEL_PATH

class ModelService:
    """
    Сервис для работы с ML моделью.
    Загружает модель, делает предсказания.
    """
    
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.model_name = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Загружает модель из файла.
        """
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")
            
            data = joblib.load(MODEL_PATH)
            
            # Проверяем структуру загруженных данных
            if isinstance(data, dict):
                self.model = data.get('model')
                self.feature_cols = data.get('feature_cols')
                self.model_name = data.get('config', {}).get('model_name', 'xgboost_model')
            else:
                # Если модель сохранена как единый объект
                self.model = data
                self.feature_cols = None
                self.model_name = 'xgboost_model'
            
            print(f"[OK] Модель загружена: {self.model_name}")
            if self.feature_cols:
                print(f"[DATA] Ожидаемое число признаков: {len(self.feature_cols)}")
            
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки модели: {e}")
            self.model = None
    
    def is_loaded(self) -> bool:
        """
        Проверяет, загружена ли модель.
        """
        return self.model is not None
    
    def predict(self, features: list) -> float:
        """
        Делает предсказание на основе входных признаков.
        """
        if not self.is_loaded():
            raise RuntimeError("Модель не загружена")
        
        try:
            # Преобразуем входные данные в numpy массив
            X = np.array(features).reshape(1, -1)
            
            # Делаем предсказание
            prediction = self.model.predict(X)[0]
            
            return float(prediction)
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при предсказании: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о модели.
        """
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded(),
            "expected_features": len(self.feature_cols) if self.feature_cols else None
        }

# Создаём глобальный экземпляр сервиса (будет использоваться приложением)
model_service = ModelService()