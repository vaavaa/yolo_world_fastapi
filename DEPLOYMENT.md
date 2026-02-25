# Инструкции по развертыванию

## Автоматическая загрузка моделей из DVC

Данный сервис автоматически загружает модели из DVC при запуске, что упрощает развертывание на новых серверах.

## Варианты развертывания

### 1. Docker Compose (рекомендуется)

#### Подготовка

1. **Клонируйте репозиторий**:
   ```bash
   git clone <URL репозитория>
   cd yolo_world_fastapi
   ```

2. **Создайте файл .env** (опционально):
   ```bash
   cp env.example .env
   # Отредактируйте .env при необходимости
   ```

3. **Создайте папки для монтирования**:
   ```bash
   ./setup_docker.sh
   ```

#### Запуск

```bash
docker-compose up --build
```

При первом запуске:
- DVC будет автоматически инициализирован
- Модели будут загружены из DVC
- Сервис будет готов к работе

### 2. Docker (без docker-compose)

```bash
# Сборка образа
docker build -t yolo-world-fastapi .

# Запуск контейнера
docker run -p 8001:8001 \
  -v $(pwd)/checkpoints:/app/src/checkpoints \
  -v $(pwd)/.dvc:/app/src/.dvc \
  yolo-world-fastapi
```

### 3. Локальное развертывание

#### Требования

- Python 3.10+
- Poetry
- DVC
- SSH доступ к remote хранилищу

#### Установка

```bash
# Клонирование и установка зависимостей
git clone <URL репозитория>
cd yolo_world_fastapi
poetry install

# Инициализация DVC
python -m yolo_world_fastapi.scripts.init_dvc

# Запуск сервиса
poetry run python -m yolo_world_fastapi
```

## Настройка DVC

### Remote хранилище

Настройки DVC задаются через переменные окружения:

```bash
# В файле .env
YOLO_WORLD_FASTAPI_DVC_REMOTE_URL=ssh://dvc@192.168.0.172:222
YOLO_WORLD_FASTAPI_DVC_REMOTE_NAME=myremote
YOLO_WORLD_FASTAPI_DVC_SSH_KEY_PATH=/path/to/ssh/key
```

### Значения по умолчанию

- `DVC_REMOTE_URL`: `ssh://dvc@192.168.0.172:222`
- `DVC_REMOTE_NAME`: `myremote`
- `DVC_SSH_KEY_PATH`: `None` (используется ssh-agent)

### Изменение remote

1. **Через переменные окружения** (рекомендуется):
   ```bash
   export YOLO_WORLD_FASTAPI_DVC_REMOTE_URL=s3://my-bucket/dvc
   export YOLO_WORLD_FASTAPI_DVC_REMOTE_NAME=myremote
   ```

2. **В файле .env**:
   ```bash
   YOLO_WORLD_FASTAPI_DVC_REMOTE_URL=s3://my-bucket/dvc
   YOLO_WORLD_FASTAPI_DVC_REMOTE_NAME=myremote
   ```

3. **В docker-compose.yml**:
   ```yaml
   environment:
     YOLO_WORLD_FASTAPI_DVC_REMOTE_URL: s3://my-bucket/dvc
     YOLO_WORLD_FASTAPI_DVC_REMOTE_NAME: myremote
   ```

4. **Или настройте вручную**:
   ```bash
   dvc remote add -d myremote <ваш_remote_url>
   ```

### SSH ключи

Для SSH remote убедитесь, что:

1. **SSH ключ добавлен в ssh-agent**:
   ```bash
   ssh-add ~/.ssh/id_rsa
   ```

2. **Или укажите путь к ключу через переменную окружения**:
   ```bash
   export YOLO_WORLD_FASTAPI_DVC_SSH_KEY_PATH=/path/to/ssh/key
   ```

3. **Или в файле .env**:
   ```bash
   YOLO_WORLD_FASTAPI_DVC_SSH_KEY_PATH=/path/to/ssh/key
   ```

4. **Или настройте вручную**:
   ```bash
   dvc remote modify myremote keyfile /путь/к/вашему/ключу
   ```

## Мониторинг

### Проверка статуса

```bash
# Проверка здоровья сервиса
curl http://localhost:8001/api/v1/health

# Проверка статуса моделей
curl http://localhost:8001/api/v1/monitoring/models/status
```

### Обновление моделей

```bash
# Обновление моделей из DVC
curl -X POST http://localhost:8001/api/v1/monitoring/models/update
```

### Логи

```bash
# Просмотр логов Docker контейнера
docker-compose logs -f api

# Или для отдельного контейнера
docker logs -f <container_id>
```

## Устранение неполадок

### Модели не загружаются

1. **Проверьте настройки DVC**:
   ```bash
   docker-compose exec api dvc remote list
   docker-compose exec api dvc status
   ```

2. **Проверьте переменные окружения**:
   ```bash
   docker-compose exec api env | grep YOLO_WORLD_FASTAPI_DVC
   ```

3. **Проверьте доступ к remote**:
   ```bash
   docker-compose exec api dvc pull --dry-run
   ```

4. **Проверьте SSH ключи**:
   ```bash
   # Если используется SSH remote
   docker-compose exec api ssh -p 222 dvc@192.168.0.172
   ```

### Ошибки при запуске

1. **Проверьте логи**:
   ```bash
   docker-compose logs api
   ```

2. **Проверьте монтирование томов**:
   ```bash
   docker-compose exec api ls -la /app/src/checkpoints
   docker-compose exec api ls -la /app/src/.dvc
   ```

3. **Попробуйте ручную инициализацию**:
   ```bash
   docker-compose exec api python -m yolo_world_fastapi.scripts.init_dvc
   ```

### Проблемы с правами доступа

```bash
# Исправьте права на папки
sudo chown -R $USER:$USER checkpoints
sudo chown -R $USER:$USER .dvc
```

## Production развертывание

### Kubernetes

Используйте файлы из папки `k8s/`:

```bash
kubectl apply -f k8s/
```

### Обновление моделей в production

1. **Через API** (если доступен):
   ```bash
   curl -X POST https://your-domain.com/api/v1/monitoring/models/update
   ```

2. **Через kubectl**:
   ```bash
   kubectl exec -it deployment/yolo-world-fastapi -- python -m yolo_world_fastapi.scripts.test_models
   ```

3. **Перезапуск с новой версией**:
   ```bash
   kubectl rollout restart deployment/yolo-world-fastapi
   ```

## Безопасность

### Переменные окружения

- Не храните SSH ключи в коде
- Используйте секреты Kubernetes для production
- Настройте правильные права доступа к папкам

### Сетевая безопасность

- Ограничьте доступ к API endpoints
- Используйте HTTPS в production
- Настройте firewall правила

## Резервное копирование

### Модели

Модели хранятся в DVC, поэтому резервное копирование не требуется.

### Конфигурация

```bash
# Резервное копирование конфигурации
tar -czf config_backup.tar.gz .dvc/ .env docker-compose.yml
```

## Масштабирование

### Горизонтальное масштабирование

```bash
# Увеличьте количество реплик
kubectl scale deployment yolo-world-fastapi --replicas=3
```

### Вертикальное масштабирование

```bash
# Увеличьте ресурсы в k8s/deployment.yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
``` 
