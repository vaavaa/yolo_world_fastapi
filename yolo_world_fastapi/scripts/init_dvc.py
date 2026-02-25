#!/usr/bin/env python3
"""
Скрипт для инициализации DVC при первом запуске.
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

# Настраиваем логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Возвращает путь к корневой папке проекта."""
    current_file = Path(__file__)
    return current_file.parent.parent.parent


def init_git():
    """Инициализирует git репозиторий если его нет."""
    project_root = get_project_root()

    logger.info(f"Проверка git репозитория в папке: {project_root}")

    try:
        # Переходим в корневую папку проекта
        os.chdir(project_root)

        # Проверяем, инициализирован ли уже git
        if (project_root / ".git").exists():
            logger.info("Git репозиторий уже инициализирован")
            return True

        # Инициализируем git
        result = subprocess.run(
            ["git", "init"],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            logger.info("Git репозиторий успешно инициализирован")

            # Настраиваем базовые git настройки
            subprocess.run(["git", "config", "user.name", "DVC User"], timeout=30)
            subprocess.run(["git", "config", "user.email", "dvc@example.com"], timeout=30)

            return True
        else:
            logger.error(f"Ошибка при инициализации git: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Инициализация git превысила таймаут")
        return False
    except Exception as e:
        logger.error(f"Ошибка при инициализации git: {e}")
        return False


def init_dvc():
    """Инициализирует DVC в проекте."""
    project_root = get_project_root()

    logger.info(f"Инициализация DVC в папке: {project_root}")

    try:
        # Переходим в корневую папку проекта
        os.chdir(project_root)

        # Проверяем, инициализирован ли уже DVC
        if (project_root / ".dvc").exists():
            logger.info("DVC уже инициализирован")
            return True

        # Инициализируем DVC
        result = subprocess.run(
            ["dvc", "init"],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            logger.info("DVC успешно инициализирован")
            return True
        else:
            logger.error(f"Ошибка при инициализации DVC: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Инициализация DVC превысила таймаут")
        return False
    except Exception as e:
        logger.error(f"Ошибка при инициализации DVC: {e}")
        return False


def setup_remote():
    """Настраивает remote для DVC."""
    project_root = get_project_root()

    logger.info("Настройка remote для DVC...")

    try:
        os.chdir(project_root)

        # Импортируем настройки
        from yolo_world_fastapi.settings import settings

        remote_name = settings.dvc_remote_name
        remote_url = settings.dvc_remote_url
        ssh_key_path = settings.dvc_ssh_key_path

        logger.info(f"Используем remote: {remote_name} -> {remote_url}")

        # Проверяем, настроен ли уже remote
        result = subprocess.run(
            ["dvc", "remote", "list"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if remote_name in result.stdout:
            logger.info(f"Remote '{remote_name}' уже настроен")

            # Проверяем, совпадает ли URL
            remote_url_result = subprocess.run(
                ["dvc", "remote", "get-url", remote_name],
                capture_output=True,
                text=True,
                timeout=30
            )

            if remote_url_result.returncode == 0 and remote_url_result.stdout.strip() == remote_url:
                logger.info("Remote URL совпадает с настройками")
                return True
            else:
                logger.info("Remote URL отличается, обновляем...")
                # Удаляем старый remote
                subprocess.run(["dvc", "remote", "remove", remote_name], timeout=30)

        # Настраиваем remote
        remote_result = subprocess.run(
            ["dvc", "remote", "add", "-d", remote_name, remote_url],
            capture_output=True,
            text=True,
            timeout=30
        )

        if remote_result.returncode == 0:
            logger.info(f"Remote '{remote_name}' успешно настроен")

            # Настраиваем SSH ключ если указан
            if ssh_key_path:
                logger.info(f"Настраиваем SSH ключ: {ssh_key_path}")
                key_result = subprocess.run(
                    ["dvc", "remote", "modify", remote_name, "keyfile", ssh_key_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if key_result.returncode == 0:
                    logger.info("SSH ключ успешно настроен")
                else:
                    logger.warning(f"Не удалось настроить SSH ключ: {key_result.stderr}")

            return True
        else:
            logger.error(f"Ошибка при настройке remote: {remote_result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Настройка remote превысила таймаут")
        return False
    except Exception as e:
        logger.error(f"Ошибка при настройке remote: {e}")
        return False


def setup_credentials():
    """Настраивает креденшелы для DVC remote."""
    project_root = get_project_root()

    logger.info("Настройка креденшелов для DVC...")

    try:
        os.chdir(project_root)

        # Импортируем настройки
        from yolo_world_fastapi.settings import settings

        remote_name = settings.dvc_remote_name

        # Создаем директорию .dvc если её нет
        dvc_dir = project_root / ".dvc"
        dvc_dir.mkdir(exist_ok=True)

        # Создаем файл config.local с креденшелами
        config_local_path = dvc_dir / "config.local"

        # Получаем пароль из настроек
        password = settings.dvc_password

        config_content = f"""['remote "{remote_name}"']
    password = {password}
"""

        with open(config_local_path, 'w') as f:
            f.write(config_content)

        logger.info(f"Креденшелы созданы в файле: {config_local_path}")
        return True

    except Exception as e:
        logger.error(f"Ошибка при настройке креденшелов: {e}")
        return False


def main():
    """Основная функция."""
    logger.info("Начинаем инициализацию git и DVC...")

    # Инициализируем git
    if not init_git():
        logger.error("Не удалось инициализировать git")
        sys.exit(1)

    # Инициализируем DVC
    if not init_dvc():
        logger.error("Не удалось инициализировать DVC")
        sys.exit(1)

    # Настраиваем remote
    if not setup_remote():
        logger.error("Не удалось настроить remote для DVC")
        sys.exit(1)

    # Настраиваем креденшелы
    if not setup_credentials():
        logger.error("Не удалось настроить креденшелы для DVC")
        sys.exit(1)

    logger.info("Git и DVC успешно настроены!")
    logger.info("Теперь можно запускать сервис - модели будут автоматически загружены из DVC")


if __name__ == "__main__":
    main()
