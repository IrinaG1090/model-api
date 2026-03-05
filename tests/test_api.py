import pytest
from fastapi.testclient import TestClient
import sys
import os

# Добавляем путь к корневой папке проекта для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.services import model_service

client = TestClient(app)

def test_root_endpoint():
    """Проверяет корневой эндпоинт"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_endpoint():
    """Проверяет эндпоинт здоровья"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "api_version" in data

def test_predict_endpoint_invalid_data():
    """
    Проверяет эндпоинт предсказания с некорректными данными.
    В CI-окружении без модели тест пропускается.
    """
    # Если модель не загружена — пропускаем тест
    if not model_service.is_loaded():
        pytest.skip("Модель не загружена в CI, пропускаем тест")
        return
    
    response = client.post("/predict", json={"features": []})
    # 422 - ошибка валидации Pydantic, 400 - если сервис сам возвращает 400
    assert response.status_code in [400, 422]

def test_predict_endpoint_no_data():
    """Проверяет эндпоинт предсказания без данных"""
    response = client.post("/predict", json={})
    assert response.status_code == 422  # 422 - ошибка валидации Pydantic

def test_predict_endpoint_valid_data():
    """
    Проверяет эндпоинт предсказания с корректными данными.
    В CI-окружении без модели тест пропускается.
    """
    if not model_service.is_loaded():
        pytest.skip("Модель не загружена в CI, пропускаем тест")
        return
    
    # Создаём список из 19 признаков (как ожидает модель)
    features = [float(i) for i in range(1, 20)]
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "model_used" in data
    assert "features_count" in data
    assert data["features_count"] == 19