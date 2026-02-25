FROM python:3.12-slim-bookworm AS builder

FROM builder AS prod
RUN pip install onnxruntime
RUN pip install poetry==1.8.2

# Устанавливаем curl для health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Configuring poetry
RUN poetry config virtualenvs.create false
RUN poetry config cache-dir /tmp/poetry_cache

# Copying requirements of a project
COPY pyproject.toml poetry.lock /app/src/
WORKDIR /app/src

# Installing requirements
RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --only main

# Copying actuall application
COPY . /app/src/
RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --only main

# Создаем директорию для моделей
RUN mkdir -p /app/src/checkpoints

# Настраиваем переменные окружения для небуферизованного вывода логов
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Запускаем приложение напрямую (MinIO загрузка происходит автоматически)
CMD ["python3", "-u", "-m", "yolo_world_fastapi"]
