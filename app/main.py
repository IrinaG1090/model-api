# app/main.py
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.config import API_TITLE, API_VERSION, API_DESCRIPTION
from app.models import (
    PredictionRequest, PredictionResponse, 
    HealthResponse, ErrorResponse
)
from app.services import model_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управляет жизненным циклом приложения.
    Выполняется при старте и завершении.
    """
    # Startup
    print("[INIT] Запуск API сервиса...")
    if model_service.is_loaded():
        print("[OK] Модель успешно загружена при старте")
    else:
        print("[WARN] Модель не загружена. Сервис будет работать в ограниченном режиме")
    
    yield
    
    # Shutdown
    print("[SHUTDOWN] Завершение работы API сервиса")

# Создаём FastAPI приложение
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/", 
         response_class=JSONResponse,
         summary="Корневой эндпоинт")
async def root():
    """
    Приветственный эндпоинт.
    Возвращает базовую информацию о сервисе.
    """
    return {
        "message": "ML Model API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", 
         response_model=HealthResponse,
         summary="Проверка здоровья сервиса")
async def health_check():
    """
    Эндпоинт для проверки состояния сервиса и модели.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_service.is_loaded(),
        api_version=API_VERSION
    )

@app.post("/predict",
          response_model=PredictionResponse,
          responses={
              400: {"model": ErrorResponse},
              500: {"model": ErrorResponse}
          },
          summary="Получить предсказание")
async def predict(request: PredictionRequest):
    """
    Эндпоинт для получения предсказания от модели.
    
    - **features**: список числовых признаков (обязательно)
    
    Возвращает предсказанное значение и метаданные.
    """
    if not model_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена. Сервис временно недоступен"
        )
    
    try:
        prediction = model_service.predict(request.features)
        
        return PredictionResponse(
            prediction=prediction,
            model_used=model_service.get_model_info()["model_name"],
            features_count=len(request.features)
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ошибка в данных: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера: {e}"
        )

@app.get("/model-info",
         summary="Информация о модели")
async def model_info():
    """
    Возвращает информацию о загруженной модели.
    """
    return model_service.get_model_info()