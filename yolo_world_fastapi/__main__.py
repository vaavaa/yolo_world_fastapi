import uvicorn

from yolo_world_fastapi.gunicorn_runner import GunicornApplication
from yolo_world_fastapi.settings import settings


def main() -> None:
    """Entrypoint of the application."""
    if settings.reload:
        uvicorn.run(
            "yolo_world_fastapi.web.application:get_app",
            workers=settings.workers_count,
            host=settings.host,
            port=settings.port,
            reload=settings.reload,
            log_level=settings.log_level.value.lower(),
            factory=True,
        )
        
    else:
        # We choose gunicorn only if reload
        # option is not used, because reload
        # feature doesn't work with gunicorn workers.
        GunicornApplication(
            "yolo_world_fastapi.web.application:get_app",
            host=settings.host,
            port=settings.port,
            workers=settings.workers_count,
            factory=True,
            accesslog="-",
            errorlog="-",
            loglevel=settings.log_level.value.lower(),
            access_log_format='%r "-" %s "-" %Tf',
            capture_output=True,
            enable_stdio_inheritance=True,
            timeout=300,  # Увеличиваем timeout до 5 минут для загрузки моделей
            graceful_timeout=60,  # Время на graceful shutdown
        ).run()


if __name__ == "__main__":
    main()
