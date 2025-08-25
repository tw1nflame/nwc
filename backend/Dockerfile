FROM python:3.11-slim

# Создаем корневую директорию для всего кода
WORKDIR /code

# Копируем сначала requirements.txt для кэширования слоев Docker
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь остальной код из контекста (из папки backend)
# Теперь у нас будет структура /code/app/main.py, /code/config_refined.yaml и т.д.
COPY . .

# <--- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ
# Меняем рабочую директорию на ту, где лежит наше приложение.
# Все последующие команды (CMD, ENTRYPOINT, RUN) будут выполняться из /code/app
WORKDIR /code/app

# Теперь эта команда будет запущена из /code/app и легко найдет main.py, tasks.py и utils/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]