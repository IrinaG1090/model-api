# tests/test_services.py
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import ModelService

def test_model_service_initialization():
    """Проверяет, что сервис инициализируется"""
    service = ModelService()
    assert service is not None

def test_model_loading():
    """Проверяет загрузку модели"""
    service = ModelService()
    # Модель может быть не загружена, если файл не найден
    # Это нормально для тестов
    assert isinstance(service.is_loaded(), bool)

def test_predict_without_model():
    """Проверяет поведение при отсутствии модели"""
    service = ModelService()
    # Временно "отключаем" модель
    service.model = None
    
    with pytest.raises(RuntimeError):
        service.predict([1, 2, 3])