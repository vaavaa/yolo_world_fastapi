"""Production конфигурация Gunicorn для максимальной производительности."""

import multiprocessing
import os

# Количество воркеров
workers = int(os.getenv("YOLO_WORLD_FASTAPI_WORKERS_COUNT", "4"))

# Биндинг
bind = f"0.0.0.0:{os.getenv('YOLO_WORLD_FASTAPI_PORT', '8000')}"

# Класс воркера
worker_class = "yolo_world_fastapi.gunicorn_runner.UvicornWorker"

# Таймауты
timeout = 120  # Для ML операций
keepalive = 5
graceful_timeout = 30

# Перезапуск воркеров
max_requests = 1000
max_requests_jitter = 100

# Предзагрузка приложения
preload_app = True

# Временные файлы в RAM
worker_tmp_dir = "/dev/shm"

# Соединения
worker_connections = 1000

# Логирование
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Безопасность
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Процессы
worker_processes = workers
