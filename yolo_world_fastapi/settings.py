import enum
from pathlib import Path
from tempfile import gettempdir

from pydantic_settings import BaseSettings, SettingsConfigDict

TEMP_DIR = Path(gettempdir())


class LogLevel(str, enum.Enum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    # quantity of workers for uvicorn
    workers_count: int = 1
    # Enable uvicorn reloading
    reload: bool = False

    # Current environment
    environment: str = "dev"

    log_level: LogLevel = LogLevel.INFO

    label_classes: list[str] = ["electronics","furniture","toys","sports","tools","decor","health","computers","hygiene","other","merch","accessories","item","dishes","lights","carpet","plant", "dress", "shoe","pants"]
    iou_threshold: float = 0.5
    score_threshold: float = 0.20
    max_num_detections: int = 10
    min_area_ratio: float = 0.0005  # Минимальный коэффициент площади детекции относительно общей площади изображения

    # Настройки качества обработки изображений
    image_interpolation: str = "cubic"  # Алгоритм интерполяции: "linear", "cubic", "lanczos"
    image_quality_mode: str = "balanced"  # Режим качества: "fast", "balanced", "high_quality"

    circle_radius_percent: float = 0.04
    text_scale: int = 2
    text_thickness: int = 3

    # MinIO settings
    minio_endpoint: str = "cdn0.mysite.kz"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "yolo-models"
    minio_secure: bool = True
    skip_model_download: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="YOLO_WORLD_FASTAPI_",
        env_file_encoding="utf-8",
    )


settings = Settings()
