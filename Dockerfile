# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Устанавливаем системные зависимости (если нужны)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем Python-пакеты
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY app/ ./app/
COPY models/ ./models/

# Указываем порт
EXPOSE 8000

# Команда для запуска
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]