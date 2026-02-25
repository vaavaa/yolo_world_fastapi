#!/usr/bin/env python3
"""
Скрипт для тестирования загрузки моделей.
"""
import asyncio
import logging
import sys
from pathlib import Path

# Настраиваем логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Возвращает путь к корневой папке проекта."""
    current_file = Path(__file__)
    return current_file.parent.parent.parent


async def test_model_loading():
    """Тестирует загрузку моделей."""
    try:
        from yolo_world_fastapi.services.model_manager import ModelManager
        
        project_root = get_project_root()
        logger.info(f"Тестирование загрузки моделей в: {project_root}")
        
        model_manager = ModelManager(str(project_root))
        
        # Проверяем статус моделей
        models_exist = model_manager._check_models_exist()
        logger.info(f"Модели присутствуют: {models_exist}")
        
        if not models_exist:
            logger.info("Попытка загрузки моделей из DVC...")
            success = await model_manager.ensure_models_available()
            
            if success:
                logger.info("✅ Модели успешно загружены!")
            else:
                logger.error("❌ Не удалось загрузить модели")
                return False
        else:
            logger.info("✅ Модели уже присутствуют")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании: {e}")
        return False


async def test_model_update():
    """Тестирует обновление моделей."""
    try:
        from yolo_world_fastapi.services.model_manager import ModelManager
        
        project_root = get_project_root()
        logger.info(f"Тестирование обновления моделей в: {project_root}")
        
        model_manager = ModelManager(str(project_root))
        
        # Пытаемся обновить модели
        logger.info("Попытка обновления моделей из DVC...")
        success = await model_manager.update_models()
        
        if success:
            logger.info("✅ Модели успешно обновлены!")
        else:
            logger.warning("⚠️ Не удалось обновить модели (возможно, они уже актуальны)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при обновлении: {e}")
        return False


async def main():
    """Основная функция."""
    logger.info("🚀 Начинаем тестирование загрузки моделей...")
    
    # Тестируем загрузку
    load_success = await test_model_loading()
    
    if load_success:
        # Тестируем обновление
        await test_model_update()
    
    logger.info("🏁 Тестирование завершено")


if __name__ == "__main__":
    asyncio.run(main()) 