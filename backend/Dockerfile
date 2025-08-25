
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё содержимое app внутрь /app (без вложения app/app)
COPY ./app/* ./
COPY ../config_refined.yaml ./config_refined.yaml

# Запуск только приложения, celery запускается отдельным процессом через docker-compose
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
