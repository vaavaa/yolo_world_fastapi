# YOLO World FastAPI

REST-сервис для детекции объектов на изображениях с помощью модели **YOLO World v2** (ONNX). Реализован на FastAPI с автоматической загрузкой моделей из MinIO.

## Особенности

- **Детекция по произвольным классам** — передача списка текстовых меток и порогов (IoU, score)
- **Два формата ответа**: изображение с разметкой (pre-stream) или JSON с боксами и опционально base64-изображением
- **Автоматическая загрузка моделей** — при старте сервис проверяет наличие моделей и при необходимости загружает их из MinIO
- **API управления моделями** — проверка статуса и обновление моделей через REST
- **Мониторинг детекций** — статистика запросов, эффективность по классам и рекомендации по настройке

## Требования

- Python 3.10+
- Poetry

## Установка и запуск

### Локальная разработка

1. **Клонируйте репозиторий**:
   ```bash
   git clone <URL репозитория>
   cd yolo_world_fastapi
   ```

2. **Установите зависимости**:
   ```bash
   poetry install
   ```

3. **Настройте окружение** (MinIO и при необходимости порт):
   ```bash
   cp env.example .env
   # Отредактируйте .env: MINIO_ACCESS_KEY, MINIO_SECRET_KEY и др.
   ```

4. **Запустите сервис**:
   ```bash
   poetry run python -m yolo_world_fastapi
   ```

   По умолчанию приложение доступно на `http://127.0.0.1:8000`. Порт и хост задаются в `.env` или переменными `YOLO_WORLD_FASTAPI_PORT`, `YOLO_WORLD_FASTAPI_HOST`.

### Docker

```bash
# Подготовка папок (checkpoints, .dvc, .git)
./setup_docker.sh

# Сборка и запуск через docker-compose (порт 8001)
docker-compose up --build
```

Или без docker-compose:

```bash
docker build -t yolo-world-fastapi .
docker run -p 8001:8001 \
  -v $(pwd)/checkpoints:/app/src/checkpoints \
  -e YOLO_WORLD_FASTAPI_MINIO_ACCESS_KEY=... \
  -e YOLO_WORLD_FASTAPI_MINIO_SECRET_KEY=... \
  yolo-world-fastapi
```

При первом запуске в контейнере модели автоматически загружаются из MinIO в `checkpoints/` (если не включён `YOLO_WORLD_FASTAPI_SKIP_MODEL_DOWNLOAD=True`).

Подробнее: [DEPLOYMENT.md](DEPLOYMENT.md).

## API

Базовый префикс: **`/api/v1`**. Интерактивная документация: **`/docs`** (Swagger UI).

### Детекция (YOLO World)

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/v1/yworld/pre-stream` | Изображение → изображение с нарисованными боксами (PNG) |
| `POST` | `/api/v1/yworld/base64` | Изображение → JSON с боксами и опционально полем `img_base64` |

Тело запроса (multipart):

- `img_stream` — файл изображения (обязательно)
- `class_names` — список строк-классов (опционально, по умолчанию из настроек)
- `iou_threshold`, `score_threshold`, `max_num_detections` — опционально

### Здоровье и модели

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/api/v1/health` | Состояние сервиса и загруженных моделей (yolo_world, nms, textual) |
| `GET` | `/api/v1/models/status` | Наличие файлов моделей в `checkpoints/` |
| `POST` | `/api/v1/models/update` | Обновление моделей из MinIO (перезагрузка) |

### Мониторинг детекций

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/api/v1/detection-stats` | Полная статистика детекций и эффективность по классам |
| `GET` | `/api/v1/detection-stats/summary` | Краткая сводка (запросы, детекции, топ/слабые классы) |
| `GET` | `/api/v1/detection-stats/classes` | Детальная статистика по каждому классу |
| `GET` | `/api/v1/detection-stats/recommendations` | Рекомендации по оптимизации классов |
| `POST` | `/api/v1/detection-stats/reset` | Сброс всей статистики детекций |

### Примеры запросов

```bash
# Проверка здоровья (порт 8001 при запуске через Docker)
curl http://localhost:8001/api/v1/health

# Статус моделей
curl http://localhost:8001/api/v1/models/status

# Обновление моделей из MinIO
curl -X POST http://localhost:8001/api/v1/models/update

# Детекция: загрузка изображения и получение картинки с боксами
curl -X POST http://localhost:8001/api/v1/yworld/pre-stream \
  -F "img_stream=@image.jpg" \
  -o result.png
```

## Модели

### Автоматическая загрузка из MinIO

При старте сервис:

1. Проверяет наличие нужных файлов в `checkpoints/`
2. При отсутствии — загружает их из настроенного MinIO bucket
3. Логирует процесс загрузки

Загрузку можно отключить (например, если модели уже смонтированы в контейнер):

```bash
YOLO_WORLD_FASTAPI_SKIP_MODEL_DOWNLOAD=True
```

### Необходимые файлы в `checkpoints/`

- `yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx`
- `non_maximum_suppression.onnx`
- `vitb32-textual.onnx`
- `yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth`

## Настройка MinIO

Переменные окружения (префикс `YOLO_WORLD_FASTAPI_`):

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `MINIO_ENDPOINT` | Хост MinIO (или S3-совместимый) | `minio.example.com` |
| `MINIO_ACCESS_KEY` | Access key | — |
| `MINIO_SECRET_KEY` | Secret key | — |
| `MINIO_BUCKET` | Имя bucket с моделями | `yolo-models` |
| `MINIO_SECURE` | Использовать HTTPS | `True` |
| `SKIP_MODEL_DOWNLOAD` | Не загружать модели при старте | `False` |

Пример в `.env`:

```bash
YOLO_WORLD_FASTAPI_MINIO_ENDPOINT=minio.example.com
YOLO_WORLD_FASTAPI_MINIO_ACCESS_KEY=your_access_key
YOLO_WORLD_FASTAPI_MINIO_SECRET_KEY=your_secret_key
YOLO_WORLD_FASTAPI_MINIO_BUCKET=yolo-models
YOLO_WORLD_FASTAPI_MINIO_SECURE=True
```

В `docker-compose.yml` эти переменные можно передать через `environment` или `env_file: .env`.

## Прочие настройки

В `env.example` и в коде (см. `yolo_world_fastapi/settings.py`) задаются:

- **Сервер**: `HOST`, `PORT` (по умолчанию 8000 локально; в примерах Docker — 8001), `WORKERS_COUNT`, `RELOAD`
- **Детекция**: `LABEL_CLASSES`, `IOU_THRESHOLD`, `SCORE_THRESHOLD`, `MAX_NUM_DETECTIONS`, `MIN_AREA_RATIO`
- **Качество изображений**: `IMAGE_INTERPOLATION`, `IMAGE_QUALITY_MODE`
- **Визуализация**: `CIRCLE_RADIUS_PERCENT`, `TEXT_SCALE`, `TEXT_THICKNESS`

## Устранение неполадок

### Модели не загружаются из MinIO

1. Проверьте переменные MinIO и доступность хоста/порта (в т.ч. HTTPS).
2. Убедитесь, что в bucket лежат все четыре файла из списка выше (имена совпадают).
3. При использовании самоподписанного сертификата проверьте настройки SSL в коде MinIO-клиента при необходимости.

### Сервис не стартует или падает при загрузке моделей

1. Проверьте логи контейнера/процесса.
2. Убедитесь, что каталог `checkpoints` доступен на запись (в Docker — см. volume в `docker-compose.yml`).
3. Для отладки можно временно включить `SKIP_MODEL_DOWNLOAD=True` и положить модели вручную.

### DVC (опционально)

В проекте сохранена поддержка загрузки моделей через DVC (скрипт `init_dvc`, модуль `model_manager`). В текущем режиме работы приложение использует **MinIO** (см. `MinIOModelManager` в `lifespan.py`). Если нужен сценарий с DVC, настройте remote и переменные DVC отдельно (см. `scripts/init_dvc.py` и при необходимости добавьте соответствующие поля в `settings`).

## Разработка

### Запуск тестов

```bash
pytest tests/
```

### Добавление новой модели в MinIO

1. Положите файл в bucket (тот же префикс/имя, что ожидается в `checkpoints/`).
2. При необходимости обновите список `required_files` в `yolo_world_fastapi/services/minio_model_manager.py`.
3. Перезапустите сервис или вызовите `POST /api/v1/models/update`.

## Лицензия

[Укажите лицензию проекта]
