
FROM python:3.11-slim

WORKDIR /app

# Копируем requirements.txt, .env и config_refined.yaml из backend
COPY requirements.txt ./
COPY .env ./
COPY config_refined.yaml ./

RUN pip install --no-cache-dir -r requirements.txt


# Копируем всё содержимое app внутрь /app (main.py, tasks.py, utils/ и т.д.)
COPY app/* ./

# Запуск только приложения, celery запускается отдельным процессом через docker-compose
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
