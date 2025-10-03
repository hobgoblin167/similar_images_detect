# Базовый образ с Python
FROM python:3.10-slim

# Устанавливаем зависимости системы (чтобы работали OpenCV и rawpy)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем код внутрь контейнера
WORKDIR /app
COPY . /app

# Ставим зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# По умолчанию выводим help
CMD ["python", "main.py", "-h"]
