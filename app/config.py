# app/config.py
import os
from pathlib import Path

# Базовые пути
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Настройки API
API_TITLE = "ML Model API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Production-ready API для модели прогнозирования временных рядов"

# Путь к модели
MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"

# Параметры предсказания
PREDICTION_TIMEOUT = 30  # секунд

# Настройки сервера
HOST = "0.0.0.0"
PORT = 8000
RELOAD = False  # True только для разработки

print(f"[OK] Конфигурация загружена")
print(f"[DATA] Модель ожидается по пути: {MODEL_PATH}")