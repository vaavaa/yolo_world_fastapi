from contextlib import asynccontextmanager
from typing import AsyncGenerator
import os
import logging
import asyncio
import subprocess
import sys

from fastapi import FastAPI
import onnxruntime
from loguru import logger

from yolo_world_fastapi.services.minio_model_manager import MinIOModelManager


def get_project_root() -> str:
    """Возвращает путь к корню проекта (где лежит checkpoints)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


async def ensure_dvc_initialized():
    """
    Обеспечивает инициализацию DVC перед загрузкой моделей.
    """
    project_root = get_project_root()

    logger.info("Проверка инициализации DVC...")

    # Проверяем, инициализирован ли DVC
    if os.path.exists(os.path.join(project_root, ".dvc")):
        logger.info("DVC уже инициализирован")
        return True

    logger.info("DVC не инициализирован. Начинаем инициализацию...")

    try:
        # Инициализируем git если нужно
        if not os.path.exists(os.path.join(project_root, ".git")):
            logger.info("Инициализация git...")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["git", "init"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            )

            if result.returncode == 0:
                logger.info("Git инициализирован успешно")
                # Настраиваем базовые git настройки
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["git", "config", "user.name", "DVC User"],
                        cwd=project_root,
                        timeout=30
                    )
                )
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["git", "config", "user.email", "dvc@example.com"],
                        cwd=project_root,
                        timeout=30
                    )
                )
            else:
                logger.warning(f"Ошибка при инициализации git: {result.stderr}")

        # Инициализируем DVC
        logger.info("Инициализация DVC...")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["dvc", "init"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
        )

        if result.returncode == 0:
            logger.info("DVC инициализирован успешно")

            # Настраиваем remote
            from yolo_world_fastapi.settings import settings

            remote_result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["dvc", "remote", "add", "-d", settings.dvc_remote_name, settings.dvc_remote_url],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            )

            if remote_result.returncode == 0:
                logger.info(f"Remote '{settings.dvc_remote_name}' настроен успешно")
                return True
            else:
                logger.warning(f"Ошибка при настройке remote: {remote_result.stderr}")
                return False
        else:
            logger.error(f"Ошибка при инициализации DVC: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Ошибка при инициализации DVC: {e}")
        return False


async def ensure_models_loaded():
    """
    Обеспечивает загрузку моделей из MinIO перед запуском приложения.

    :return: True если модели загружены, False если нет
    """
    print("=== Начинаем проверку и загрузку моделей ===", flush=True)
    logger.info("=== Начинаем проверку и загрузку моделей ===")
    
    project_root = get_project_root()
    model_manager = MinIOModelManager(project_root)

    print(f"Проект root: {project_root}", flush=True)
    logger.info(f"Проект root: {project_root}")
    
    print("Вызываем model_manager.ensure_models_available()...", flush=True)
    logger.info("Вызываем model_manager.ensure_models_available()...")
    
    # Проверяем и загружаем модели при необходимости
    models_available = await model_manager.ensure_models_available()
    
    print(f"Результат ensure_models_available: {models_available}", flush=True)
    logger.info(f"Результат ensure_models_available: {models_available}")

    if not models_available:
        print("Не удалось загрузить модели из MinIO", flush=True)
        logger.warning(
            "Не удалось загрузить модели из MinIO. "
            "Убедитесь, что MinIO настроен правильно и доступ к хранилищу работает."
        )
        return False

    print("=== Модели успешно загружены ===", flush=True)
    logger.info("=== Модели успешно загружены ===")
    return True


async def load_model():
    """
    Асинхронная загрузка модели YOLO World.

    :return: inference_session, image_size
    """
    here = get_project_root()
    onnx_file = os.path.join(
        here,
        "checkpoints/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx",  # noqa: E501
    )
    if not os.path.exists(onnx_file):
        raise FileNotFoundError(
            f"File not found: {onnx_file}, download it from "
            "https://github.com/wkentaro/yolo-world-onnx/releases/latest"
        )

    # Запускаем загрузку модели в отдельном потоке, чтобы не блокировать основной поток
    loop = asyncio.get_event_loop()
    inference_session, image_size = await loop.run_in_executor(
        None,
        lambda: (
            onnxruntime.InferenceSession(
                path_or_bytes=onnx_file,
                providers=['CPUExecutionProvider']
            ),
            640
        )
    )

    return inference_session, image_size


async def load_nms_and_textual_models():
    """
    Асинхронная загрузка NMS и textual моделей.
    """
    here = get_project_root()
    nms_onnx_path = os.path.join(here, "checkpoints/non_maximum_suppression.onnx")
    textual_onnx_path = os.path.join(here, "checkpoints/vitb32-textual.onnx")
    loop = asyncio.get_event_loop()
    nms_inference_session, textual_inference_session = await loop.run_in_executor(
        None,
        lambda: (
            onnxruntime.InferenceSession(nms_onnx_path, providers=['CPUExecutionProvider']),
            onnxruntime.InferenceSession(textual_onnx_path, providers=['CPUExecutionProvider'])
        )
    )
    return nms_inference_session, textual_inference_session


@asynccontextmanager
async def lifespan_setup(
    app: FastAPI,
) -> AsyncGenerator[None, None]:  # pragma: no cover
    """
    Actions to run on application startup.

    This function uses fastAPI app to store data
    in the state, such as db_engine.

    :param app: the fastAPI application.
    :return: function that actually performs actions.
    """
    
    # Принудительное логирование для отладки
    print("=== LIFESPAN SETUP STARTED ===", flush=True)
    logger.info("=== LIFESPAN SETUP STARTED ===")
    sys.stdout.flush()
    sys.stderr.flush()

    app.middleware_stack = None
    app.middleware_stack = app.build_middleware_stack()

    # Обеспечиваем загрузку моделей из MinIO
    from yolo_world_fastapi.settings import settings

    if settings.skip_model_download:
        print("Пропускаем загрузку моделей (SKIP_MODEL_DOWNLOAD=true)", flush=True)
        logger.info("Пропускаем загрузку моделей (SKIP_MODEL_DOWNLOAD=true)")
        models_loaded = False
    else:
        # Обеспечиваем загрузку моделей из MinIO перед запуском
        print("Проверка и загрузка моделей из MinIO...", flush=True)
        logger.info("Проверка и загрузка моделей из MinIO...")
        models_loaded = await ensure_models_loaded()
        if models_loaded:
            print("Модели из MinIO загружены успешно", flush=True)
            logger.info("Модели из MinIO загружены успешно")
        else:
            print("Модели из MinIO не загружены, но приложение продолжает работу", flush=True)
            logger.warning("Модели из MinIO не загружены, но приложение продолжает работу")
    
    sys.stdout.flush()
    sys.stderr.flush()

    # Загружаем модель YOLO World при запуске приложения
    logger.info("Загрузка модели YOLO World...")
    try:
        yolo_world_session, image_size = await load_model()
        app.state.yolo_world_session = yolo_world_session
        app.state.image_size = image_size
        logger.info(f"Модель YOLO World загружена успешно. Размер изображения: {image_size}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели YOLO World: {e}")
        app.state.yolo_world_session = None
        app.state.image_size = None

    # Загружаем NMS и textual модели
    logger.info("Загрузка NMS и textual моделей...")
    try:
        nms_inference_session, textual_inference_session = await load_nms_and_textual_models()
        app.state.nms_inference_session = nms_inference_session
        app.state.textual_inference_session = textual_inference_session
        logger.info("NMS и textual модели загружены успешно.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке NMS и textual моделей: {e}")
        app.state.nms_inference_session = None
        app.state.textual_inference_session = None

    print("=== LIFESPAN SETUP COMPLETED ===", flush=True)
    logger.info("=== LIFESPAN SETUP COMPLETED ===")
    
    yield
    
    # Shutdown логика
    print("=== LIFESPAN SHUTDOWN STARTED ===", flush=True)
    logger.info("=== LIFESPAN SHUTDOWN STARTED ===")
    
    # Очищаем ресурсы
    if hasattr(app.state, 'yolo_world_session') and app.state.yolo_world_session:
        print("Очищаем YOLO World session...", flush=True)
        logger.info("Очищаем YOLO World session...")
        del app.state.yolo_world_session
    
    if hasattr(app.state, 'nms_inference_session') and app.state.nms_inference_session:
        print("Очищаем NMS session...", flush=True)
        logger.info("Очищаем NMS session...")
        del app.state.nms_inference_session
        
    if hasattr(app.state, 'textual_inference_session') and app.state.textual_inference_session:
        print("Очищаем textual session...", flush=True)
        logger.info("Очищаем textual session...")
        del app.state.textual_inference_session
    
    print("=== LIFESPAN SHUTDOWN COMPLETED ===", flush=True)
    logger.info("=== LIFESPAN SHUTDOWN COMPLETED ===")
