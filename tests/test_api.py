# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

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
    """Проверяет эндпоинт предсказания с некорректными данными"""
    response = client.post("/predict", json={"features": []})
    assert response.status_code in [400, 422]  # 422 - ошибка валидации Pydantic

def test_predict_endpoint_no_data():
    """Проверяет эндпоинт предсказания без данных"""
    response = client.post("/predict", json={})
    assert response.status_code == 422  # 422 - ошибка валидации Pydantic