"""
Monitoring views.
"""
from fastapi import APIRouter, HTTPException
from fastapi import Request
from yolo_world_fastapi.web.api.monitoring.schema import HealthResponse
from yolo_world_fastapi.web.api.monitoring.detection_schema import MonitoringResponse
from yolo_world_fastapi.services.minio_model_manager import MinIOModelManager
from yolo_world_fastapi.services.detection_monitor import detection_monitor
from yolo_world_fastapi.web.lifespan import get_project_root
from yolo_world_fastapi.web.responses import ORJSONResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """
    Проверка здоровья сервиса.

    :return: Статус здоровья сервиса
    """
    # Получаем состояние приложения
    app = request.app
    
    # Проверяем состояние моделей
    models_status = {
        "yolo_world": getattr(app.state, 'yolo_world_session', None) is not None,
        "nms": getattr(app.state, 'nms_inference_session', None) is not None,
        "textual": getattr(app.state, 'textual_inference_session', None) is not None
    }
    
    # Определяем общий статус
    all_models_loaded = all(models_status.values())
    status = "ok" if all_models_loaded else "warning"
    
    return HealthResponse(
        status=status,
        details={
            "models": models_status,
            "message": "Все модели загружены" if all_models_loaded else "Некоторые модели не загружены"
        }
    )

@router.post("/models/update")
async def update_models() -> ORJSONResponse:
    """
    Обновляет модели из MinIO.
    
    :return: Результат обновления
    """
    try:
        project_root = get_project_root()
        model_manager = MinIOModelManager(project_root)
        
        success = await model_manager.update_models()
        
        if success:
            return ORJSONResponse(content={"status": "success", "message": "Модели успешно обновлены"})
        else:
            raise HTTPException(
                status_code=500, 
                detail="Не удалось обновить модели из DVC"
            )
            
    except Exception as e:
        logger.error(f"Ошибка при обновлении моделей: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


@router.get("/models/status")
async def check_models_status() -> ORJSONResponse:
    """
    Проверяет статус моделей.
    
    :return: Статус моделей
    """
    try:
        project_root = get_project_root()
        model_manager = MinIOModelManager(project_root)
        
        models_exist = model_manager._check_models_exist()
        
        return ORJSONResponse(content={
            "status": "available" if models_exist else "missing",
            "models_exist": models_exist
        })
        
    except Exception as e:
        logger.error(f"Ошибка при проверке статуса моделей: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


@router.get("/detection-stats", response_model=MonitoringResponse)
async def get_detection_stats() -> MonitoringResponse:
    """
    Возвращает статистику детекций и анализ эффективности классов.
    
    :return: Полная статистика мониторинга детекций
    """
    try:
        return detection_monitor.get_monitoring_data()
    except Exception as e:
        logger.error(f"Ошибка при получении статистики детекций: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


@router.get("/detection-stats/summary")
async def get_detection_summary() -> ORJSONResponse:
    """
    Возвращает краткую сводку по детекциям.
    
    :return: Краткая статистика
    """
    try:
        metrics = detection_monitor.get_metrics()
        effectiveness = detection_monitor.get_class_effectiveness()
        recommendations = detection_monitor.get_recommendations()
        
        # Топ-5 наиболее детектируемых классов
        top_classes = [e.class_name for e in effectiveness[:5]]
        
        # Топ-5 наименее эффективных классов
        bottom_classes = [e.class_name for e in effectiveness[-5:] if e.detection_rate < 10]
        
        return ORJSONResponse(content={
            "total_requests": metrics.total_requests,
            "total_detections": metrics.total_detections,
            "avg_processing_time": round(metrics.avg_processing_time, 3),
            "avg_detections_per_image": round(metrics.avg_detections_per_image, 2),
            "top_detected_classes": top_classes,
            "low_effectiveness_classes": bottom_classes,
            "recommendations_count": len(recommendations),
            "has_recommendations": len(recommendations) > 0
        })
    except Exception as e:
        logger.error(f"Ошибка при получении сводки детекций: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


@router.get("/detection-stats/classes")
async def get_class_effectiveness() -> ORJSONResponse:
    """
    Возвращает детальную статистику по эффективности классов.
    
    :return: Статистика по классам
    """
    try:
        effectiveness = detection_monitor.get_class_effectiveness()
        stats = detection_monitor.get_detection_stats()
        
        # Объединяем данные
        class_data = []
        for eff in effectiveness:
            # Находим соответствующие статистики
            stat = next((s for s in stats if s.class_name == eff.class_name), None)
            
            class_data.append({
                "class_name": eff.class_name,
                "detection_rate": round(eff.detection_rate, 2),
                "avg_confidence": round(eff.avg_confidence, 3),
                "total_requests": eff.total_requests,
                "successful_detections": eff.successful_detections,
                "total_detections": stat.total_detections if stat else 0,
                "min_confidence": round(stat.min_confidence, 3) if stat else 0.0,
                "max_confidence": round(stat.max_confidence, 3) if stat else 0.0
            })
        
        return ORJSONResponse(content={
            "classes": class_data,
            "total_classes": len(class_data),
            "active_classes": len([c for c in class_data if c["total_requests"] > 0]),
            "unused_classes": len([c for c in class_data if c["total_requests"] == 0])
        })
    except Exception as e:
        logger.error(f"Ошибка при получении статистики классов: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


@router.get("/detection-stats/recommendations")
async def get_recommendations() -> ORJSONResponse:
    """
    Возвращает рекомендации по оптимизации классов.
    
    :return: Список рекомендаций
    """
    try:
        recommendations = detection_monitor.get_recommendations()
        return ORJSONResponse(content={
            "recommendations": recommendations,
            "count": len(recommendations)
        })
    except Exception as e:
        logger.error(f"Ошибка при получении рекомендаций: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


@router.post("/detection-stats/reset")
async def reset_detection_stats() -> ORJSONResponse:
    """
    Сбрасывает всю статистику детекций.
    
    :return: Подтверждение сброса
    """
    try:
        detection_monitor.reset_stats()
        return ORJSONResponse(content={
            "status": "success",
            "message": "Статистика детекций сброшена"
        })
    except Exception as e:
        logger.error(f"Ошибка при сбросе статистики: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )

