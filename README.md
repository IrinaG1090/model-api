# Model API — Production-ready ML микросервис

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.115-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen)](https://www.docker.com/)

## 📝 О проекте

**Model API** — это production-ready микросервис для serving ML моделей. Проект является шестым этапом большого плана по созданию production-ready AI систем.

### ✨ Возможности
- 🚀 Высокопроизводительный API на FastAPI
- 📊 Автоматическая документация Swagger/ReDoc
- 🔄 Загрузка модели из файла
- 🧪 Встроенные тесты
- 🐳 Готовая Docker-контейнеризация
- 🔧 Подготовка к CI/CD (GitHub Actions)

## 🛠️ Технологический стек

| Компонент | Технология |
|:---|:---|
| **API** | FastAPI |
| **Сериализация** | Pydantic |
| **Модель** | XGBoost (из Проекта 5) |
| **Тестирование** | Pytest + httpx |
| **Контейнеризация** | Docker |

## 📋 Предварительные требования

- Python 3.12+
- Модель из Проекта 5 (или любая другая совместимая модель)
- Docker (опционально)

## 🚀 Быстрый старт

### 1. Клонировать и подготовить модель
```bash
git clone https://github.com/IrinaG1090/model-api.git
cd model-api
```
# Скопировать обученную модель в папку models/



**2. Установить зависимости**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

**3. Запустить API**
```bash
uvicorn app.main:app --reload
```

### 4. Открыть документацию

Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc

### 5. Запуск через Docker
```bash
docker build -t model-api .
docker run -p 8000:8000 model-api
```
## 📁 Структура проекта

```text
model-api/
├── app/
│   ├── main.py          # FastAPI приложение
│   ├── models.py        # Pydantic схемы
│   ├── services.py      # Бизнес-логика
│   └── config.py        # Настройки
├── tests/
│   ├── test_api.py      # Тесты API
│   └── test_services.py # Тесты сервисов
├── models/
│   └── xgboost_model.pkl # Обученная модель
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

##  📊 Примеры запросов

**Проверка здоровья**
```bash
curl http://localhost:8000/health
```

**Получить предсказание**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [25.0, 10.0, 5.0, 2.0]}'
```

**Информация о модели**
```bash
curl http://localhost:8000/model-info
```

## 🧪 Тестирование
```bash
pytest tests/ -v
```

### 5. Запусти API
```bash
uvicorn app.main:app --reload
```

## 🚀 Открой в браузере

http://127.0.0.1:8000	Приветственное сообщение
http://127.0.0.1:8000/docs	Swagger UI (документация, можно тестировать API)
http://127.0.0.1:8000/redoc	ReDoc (альтернативная документация)

## 📄 Лицензия

MIT License
