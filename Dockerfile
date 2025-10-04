# Базовый образ Python
FROM python:3.9-slim

# Рабочая директория в контейнере
WORKDIR /app

# Скопировать зависимости
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Скопировать весь проект
COPY . .

# Открыть порт
EXPOSE 5000

# Запуск приложения
CMD ["python", "app.py"]
