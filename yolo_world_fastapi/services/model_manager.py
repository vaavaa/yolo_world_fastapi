"""
Модуль для управления загрузкой моделей из DVC.
"""
import os
import logging
import subprocess
import asyncio
from pathlib import Path
from typing import Optional

from yolo_world_fastapi.settings import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Менеджер для загрузки и управления моделями из DVC."""

    def __init__(self, project_root: str):
        """
        Инициализация менеджера моделей.

        :param project_root: Путь к корневой папке проекта
        """
        self.project_root = Path(project_root)
        self.checkpoints_dir = self.project_root / "checkpoints"

    async def ensure_models_available(self) -> bool:
        """
        Проверяет наличие моделей и загружает их из DVC при необходимости.

        :return: True если модели доступны, False в случае ошибки
        """
        try:
            # Проверяем, есть ли уже модели
            if self._check_models_exist():
                logger.info("Модели уже присутствуют в папке checkpoints")
                return True

            logger.info("Модели не найдены. Начинаем загрузку из DVC...")

            # Проверяем, инициализирован ли DVC
            if not self._check_dvc_initialized():
                logger.warning("DVC не инициализирован. Пропускаем загрузку моделей.")
                return False

            # Загружаем модели из DVC
            success = await self._pull_models_from_dvc()

            if success:
                logger.info("Модели успешно загружены из DVC")
                return True
            else:
                logger.error("Не удалось загрузить модели из DVC")
                return False

        except Exception as e:
            logger.error(f"Ошибка при проверке/загрузке моделей: {e}")
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
            if not file_path.exists():
                logger.warning(f"Файл модели не найден: {file_path}")
                return False

        return True

    def _check_dvc_initialized(self) -> bool:
        """
        Проверяет, инициализирован ли DVC в проекте.

        :return: True если DVC инициализирован
        """
        dvc_dir = self.project_root / ".dvc"
        return dvc_dir.exists()

    async def _pull_models_from_dvc(self) -> bool:
        """
        Загружает модели из DVC в отдельном процессе.

        :return: True если загрузка прошла успешно
        """
        try:
            # Запускаем dvc pull в отдельном процессе
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._run_dvc_pull
            )

            return result == 0

        except Exception as e:
            logger.error(f"Ошибка при выполнении dvc pull: {e}")
            return False

    def _run_dvc_pull(self) -> int:
        """
        Выполняет команду dvc pull синхронно.

        :return: Код возврата команды
        """
        try:
            # Проверяем, что мы в корневой папке проекта
            os.chdir(self.project_root)

            # Логируем используемые настройки
            logger.info(f"Используем DVC remote: {settings.dvc_remote_name} -> {settings.dvc_remote_url}")

            # Выполняем dvc pull
            result = subprocess.run(
                ["dvc", "pull"],
                capture_output=True,
                text=True,
                timeout=300  # 5 минут таймаут
            )

            if result.returncode != 0:
                logger.error(f"dvc pull завершился с ошибкой: {result.stderr}")
            else:
                logger.info("dvc pull выполнен успешно")

            return result.returncode

        except subprocess.TimeoutExpired:
            logger.error("dvc pull превысил таймаут (5 минут)")
            return 1
        except Exception as e:
            logger.error(f"Ошибка при выполнении dvc pull: {e}")
            return 1

    async def update_models(self) -> bool:
        """
        Обновляет модели из DVC (dvc update + dvc pull).

        :return: True если обновление прошло успешно
        """
        try:
            logger.info("Начинаем обновление моделей из DVC...")

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._run_dvc_update
            )

            if result:
                logger.info("Модели успешно обновлены")
                return True
            else:
                logger.error("Не удалось обновить модели")
                return False

        except Exception as e:
            logger.error(f"Ошибка при обновлении моделей: {e}")
            return False

    def _run_dvc_update(self) -> bool:
        """
        Выполняет dvc update и dvc pull.

        :return: True если обновление прошло успешно
        """
        try:
            os.chdir(self.project_root)

            # Логируем используемые настройки
            logger.info(f"Обновляем модели из DVC remote: {settings.dvc_remote_name}")

            # Сначала обновляем метаданные
            update_result = subprocess.run(
                ["dvc", "update"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if update_result.returncode != 0:
                logger.warning(f"dvc update завершился с предупреждением: {update_result.stderr}")

            # Затем загружаем обновленные файлы
            pull_result = subprocess.run(
                ["dvc", "pull"],
                capture_output=True,
                text=True,
                timeout=300
            )

            if pull_result.returncode == 0:
                logger.info("Модели успешно обновлены")
                return True
            else:
                logger.error(f"dvc pull завершился с ошибкой: {pull_result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Операция обновления превысила таймаут")
            return False
        except Exception as e:
            logger.error(f"Ошибка при обновлении моделей: {e}")
            return False
