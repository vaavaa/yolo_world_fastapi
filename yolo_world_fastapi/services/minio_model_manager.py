"""
Модуль для управления загрузкой моделей из MinIO.
"""
import os
import logging
import asyncio
from pathlib import Path
from typing import Optional

from minio import Minio
from minio.error import S3Error
import httpx
import urllib3

from yolo_world_fastapi.settings import settings

logger = logging.getLogger(__name__)


class MinIOModelManager:
    """Менеджер для загрузки и управления моделями из MinIO."""

    def __init__(self, project_root: str):
        """
        Инициализация менеджера моделей.

        :param project_root: Путь к корневой папке проекта
        """
        self.project_root = Path(project_root)
        self.checkpoints_dir = self.project_root / "checkpoints"
        
        # Создаем директорию для моделей если её нет
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Отключаем предупреждения SSL если нужно
        if settings.minio_secure:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Инициализация MinIO клиента
        self.minio_client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            # Отключаем проверку SSL сертификата для самоподписанных сертификатов
            cert_check=False if settings.minio_secure else True
        )

    async def ensure_models_available(self) -> bool:
        """
        Проверяет наличие моделей и загружает их из MinIO при необходимости.

        :return: True если модели доступны, False в случае ошибки
        """
        try:
            print("MinIOModelManager: 🔍 Проверяем доступность моделей...", flush=True)
            logger.info("MinIOModelManager: 🔍 Проверяем доступность моделей...")
            
            # Проверяем настройку пропуска загрузки
            if settings.skip_model_download:
                print("MinIOModelManager: ⏭️  Загрузка моделей пропущена (skip_model_download=True)", flush=True)
                logger.info("MinIOModelManager: ⏭️  Загрузка моделей пропущена (skip_model_download=True)")
                return True

            # Проверяем, есть ли уже модели
            print("MinIOModelManager: Проверяем наличие локальных моделей...", flush=True)
            logger.info("MinIOModelManager: Проверяем наличие локальных моделей...")
            
            if self._check_models_exist():
                print("MinIOModelManager: ✅ Модели уже присутствуют в папке checkpoints", flush=True)
                logger.info("MinIOModelManager: ✅ Модели уже присутствуют в папке checkpoints")
                return True

            print("MinIOModelManager: 📥 Модели не найдены. Начинаем загрузку из MinIO...", flush=True)
            logger.info("MinIOModelManager: 📥 Модели не найдены. Начинаем загрузку из MinIO...")
            print(f"MinIOModelManager: 🌐 MinIO endpoint: {settings.minio_endpoint}", flush=True)
            logger.info(f"MinIOModelManager: 🌐 MinIO endpoint: {settings.minio_endpoint}")
            print(f"MinIOModelManager: 🪣 MinIO bucket: {settings.minio_bucket}", flush=True)
            logger.info(f"MinIOModelManager: 🪣 MinIO bucket: {settings.minio_bucket}")

            # Загружаем модели из MinIO
            print("MinIOModelManager: Вызываем _download_models_from_minio()...", flush=True)
            logger.info("MinIOModelManager: Вызываем _download_models_from_minio()...")
            success = await self._download_models_from_minio()

            if success:
                logger.info("🎉 Модели успешно загружены из MinIO")
                return True
            else:
                logger.error("❌ Не удалось загрузить модели из MinIO")
                return False

        except Exception as e:
            logger.error(f"💥 Ошибка при проверке/загрузке моделей: {e}")
            return False

    def _check_models_exist(self) -> bool:
        """
        Проверяет наличие необходимых файлов моделей.

        :return: True если все модели присутствуют
        """
        required_files = [
            "yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx",
            "non_maximum_suppression.onnx",
            "vitb32-textual.onnx",
            "yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
        ]

        for file_name in required_files:
            file_path = self.checkpoints_dir / file_name
            if not file_path.exists() or file_path.stat().st_size == 0:
                logger.warning(f"Файл модели не найден или пуст: {file_path}")
                return False

        return True

    async def _download_models_from_minio(self) -> bool:
        """
        Загружает модели из MinIO.

        :return: True если загрузка прошла успешно
        """
        try:
            print("MinIOModelManager: Начинаем процесс загрузки моделей из MinIO", flush=True)
            logger.info("MinIOModelManager: Начинаем процесс загрузки моделей из MinIO")
            
            # Список файлов для загрузки
            required_files = [
                "yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx",
                "non_maximum_suppression.onnx",
                "vitb32-textual.onnx",
                "yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
            ]

            print(f"MinIOModelManager: Файлы для загрузки: {required_files}", flush=True)
            logger.info(f"MinIOModelManager: Файлы для загрузки: {required_files}")

            # Проверяем доступность bucket
            print("MinIOModelManager: Проверяем доступность bucket...", flush=True)
            logger.info("MinIOModelManager: Проверяем доступность bucket...")
            
            loop = asyncio.get_event_loop()
            bucket_exists = await loop.run_in_executor(
                None,
                self._check_bucket_exists
            )
            
            print(f"MinIOModelManager: Bucket существует: {bucket_exists}", flush=True)
            logger.info(f"MinIOModelManager: Bucket существует: {bucket_exists}")
            
            if not bucket_exists:
                print(f"MinIOModelManager: Bucket '{settings.minio_bucket}' не существует или недоступен", flush=True)
                logger.error(f"MinIOModelManager: Bucket '{settings.minio_bucket}' не существует или недоступен")
                return False

            # Загружаем каждый файл последовательно (вместо параллельно для лучшего логирования)
            print("MinIOModelManager: Начинаем последовательную загрузку файлов...", flush=True)
            logger.info("MinIOModelManager: Начинаем последовательную загрузку файлов...")
            
            success_count = 0
            for i, file_name in enumerate(required_files):
                print(f"MinIOModelManager: Загружаем файл {i+1}/{len(required_files)}: {file_name}", flush=True)
                logger.info(f"MinIOModelManager: Загружаем файл {i+1}/{len(required_files)}: {file_name}")
                
                success = await self._download_single_file(file_name)
                if success:
                    success_count += 1
                    print(f"MinIOModelManager: ✅ Файл {file_name} загружен успешно ({success_count}/{len(required_files)})", flush=True)
                    logger.info(f"MinIOModelManager: ✅ Файл {file_name} загружен успешно ({success_count}/{len(required_files)})")
                else:
                    print(f"MinIOModelManager: ❌ Не удалось загрузить {file_name}", flush=True)
                    logger.error(f"MinIOModelManager: ❌ Не удалось загрузить {file_name}")

            # Проверяем итоговые результаты
            print(f"MinIOModelManager: Загружено {success_count} из {len(required_files)} файлов", flush=True)
            logger.info(f"MinIOModelManager: Загружено {success_count} из {len(required_files)} файлов")
            
            if success_count == len(required_files):
                print("MinIOModelManager: ✅ Все файлы успешно загружены", flush=True)
                logger.info("MinIOModelManager: ✅ Все файлы успешно загружены")
                return True
            else:
                print(f"MinIOModelManager: ❌ Загружено только {success_count} из {len(required_files)} файлов", flush=True)
                logger.error(f"MinIOModelManager: ❌ Загружено только {success_count} из {len(required_files)} файлов")
                return False

        except Exception as e:
            logger.error(f"Ошибка при загрузке моделей из MinIO: {e}")
            return False

    def _check_bucket_exists(self) -> bool:
        """
        Проверяет существование bucket в MinIO.

        :return: True если bucket существует
        """
        try:
            return self.minio_client.bucket_exists(settings.minio_bucket)
        except S3Error as e:
            logger.error(f"Ошибка при проверке bucket: {e}")
            return False

    def _get_remote_file_size(self, file_name: str) -> int:
        """
        Получает размер файла в MinIO.

        :param file_name: Имя файла в MinIO
        :return: Размер файла в байтах
        """
        try:
            stat_info = self.minio_client.stat_object(settings.minio_bucket, file_name)
            return stat_info.size
        except S3Error as e:
            logger.error(f"Ошибка при получении размера файла {file_name}: {e}")
            return 0
        except Exception as e:
            logger.error(f"Общая ошибка при получении размера файла {file_name}: {e}")
            return 0

    async def _download_single_file(self, file_name: str) -> bool:
        """
        Загружает один файл из MinIO.

        :param file_name: Имя файла для загрузки
        :return: True если загрузка успешна
        """
        try:
            local_path = self.checkpoints_dir / file_name
            
            print(f"MinIOModelManager: 📥 Начинаем загрузку {file_name}...", flush=True)
            logger.info(f"MinIOModelManager: 📥 Начинаем загрузку {file_name}...")
            print(f"MinIOModelManager: Локальный путь: {local_path}", flush=True)
            logger.info(f"MinIOModelManager: Локальный путь: {local_path}")
            
            # Проверяем, есть ли уже полностью загруженный файл
            if local_path.exists():
                print(f"MinIOModelManager: Локальный файл существует, проверяем размер...", flush=True)
                logger.info(f"MinIOModelManager: Локальный файл существует, проверяем размер...")
                
                try:
                    # Получаем размер файла в MinIO
                    loop = asyncio.get_event_loop()
                    remote_size = await loop.run_in_executor(
                        None,
                        self._get_remote_file_size,
                        file_name
                    )
                    
                    local_size = local_path.stat().st_size
                    print(f"MinIOModelManager: Локальный размер: {local_size}, удаленный размер: {remote_size}", flush=True)
                    logger.info(f"MinIOModelManager: Локальный размер: {local_size}, удаленный размер: {remote_size}")
                    
                    if local_size == remote_size and local_size > 0:
                        size_mb = local_size / (1024 * 1024)
                        print(f"MinIOModelManager: ✅ Файл {file_name} уже загружен полностью ({size_mb:.1f} MB)", flush=True)
                        logger.info(f"MinIOModelManager: ✅ Файл {file_name} уже загружен полностью ({size_mb:.1f} MB)")
                        return True
                    else:
                        print(f"MinIOModelManager: Размеры не совпадают, удаляем частичный файл и перезагружаем", flush=True)
                        logger.info(f"MinIOModelManager: Размеры не совпадают, удаляем частичный файл и перезагружаем")
                        local_path.unlink()
                        
                except Exception as e:
                    print(f"MinIOModelManager: Ошибка при проверке размера файла: {e}", flush=True)
                    logger.warning(f"MinIOModelManager: Ошибка при проверке размера файла: {e}")
            
            # Загружаем файл в отдельном потоке
            print(f"MinIOModelManager: Вызываем _download_file_sync для {file_name}...", flush=True)
            logger.info(f"MinIOModelManager: Вызываем _download_file_sync для {file_name}...")
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._download_file_sync,
                file_name,
                str(local_path)
            )
            
            print(f"MinIOModelManager: Загрузка {file_name} завершена, проверяем результат...", flush=True)
            logger.info(f"MinIOModelManager: Загрузка {file_name} завершена, проверяем результат...")
            
            # Проверяем, что файл загрузился
            if local_path.exists() and local_path.stat().st_size > 0:
                size_mb = local_path.stat().st_size / (1024 * 1024)
                print(f"MinIOModelManager: ✅ Файл {file_name} успешно загружен ({size_mb:.1f} MB)", flush=True)
                logger.info(f"MinIOModelManager: ✅ Файл {file_name} успешно загружен ({size_mb:.1f} MB)")
                return True
            else:
                logger.error(f"❌ Файл {file_name} не загрузился или пуст")
                return False
                
        except Exception as e:
            logger.error(f"💥 Ошибка при загрузке файла {file_name}: {e}")
            return False

    def _download_file_sync(self, file_name: str, local_path: str) -> None:
        """
        Синхронная загрузка файла из MinIO.

        :param file_name: Имя файла в MinIO
        :param local_path: Локальный путь для сохранения
        """
        try:
            self.minio_client.fget_object(
                settings.minio_bucket,
                file_name,
                local_path
            )
        except S3Error as e:
            logger.error(f"MinIO ошибка при загрузке {file_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Общая ошибка при загрузке {file_name}: {e}")
            raise

    async def update_models(self) -> bool:
        """
        Обновляет модели из MinIO (удаляет локальные и загружает заново).

        :return: True если обновление прошло успешно
        """
        try:
            logger.info("Начинаем обновление моделей из MinIO...")

            # Удаляем существующие модели
            required_files = [
                "yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx",
                "non_maximum_suppression.onnx", 
                "vitb32-textual.onnx",
                "yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
            ]

            for file_name in required_files:
                file_path = self.checkpoints_dir / file_name
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Удален старый файл: {file_name}")

            # Загружаем модели заново
            success = await self._download_models_from_minio()

            if success:
                logger.info("Модели успешно обновлены")
                return True
            else:
                logger.error("Не удалось обновить модели")
                return False

        except Exception as e:
            logger.error(f"Ошибка при обновлении моделей: {e}")
            return False

    async def list_available_models(self) -> list[str]:
        """
        Получает список доступных моделей в MinIO bucket.

        :return: Список имен файлов моделей
        """
        try:
            loop = asyncio.get_event_loop()
            objects = await loop.run_in_executor(
                None,
                self._list_objects_sync
            )
            return objects
        except Exception as e:
            logger.error(f"Ошибка при получении списка моделей: {e}")
            return []

    def _list_objects_sync(self) -> list[str]:
        """
        Синхронное получение списка объектов из MinIO.

        :return: Список имен файлов
        """
        try:
            objects = self.minio_client.list_objects(settings.minio_bucket)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            logger.error(f"MinIO ошибка при получении списка объектов: {e}")
            raise
        except Exception as e:
            logger.error(f"Общая ошибка при получении списка объектов: {e}")
            raise
