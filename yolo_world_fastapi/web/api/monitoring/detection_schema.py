from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class DetectionStats(BaseModel):
    """Статистика по детекции одного класса"""
    class_name: str
    total_detections: int = 0
    avg_confidence: float = 0.0
    min_confidence: float = 1.0
    max_confidence: float = 0.0
    detection_count: int = 0  # Количество раз когда класс был детектирован


class ClassEffectiveness(BaseModel):
    """Эффективность класса"""
    class_name: str
    detection_rate: float = 0.0  # Процент изображений где класс был найден
    avg_confidence: float = 0.0
    total_requests: int = 0
    successful_detections: int = 0


class DetectionMetrics(BaseModel):
    """Общие метрики детекции"""
    total_requests: int = 0
    total_detections: int = 0
    avg_detections_per_image: float = 0.0
    avg_processing_time: float = 0.0
    most_detected_classes: List[str] = []
    least_detected_classes: List[str] = []


class DetectionLogEntry(BaseModel):
    """Запись лога детекции"""
    timestamp: datetime
    request_id: str
    classes_requested: List[str]
    classes_detected: List[str]
    confidences: List[float]
    processing_time: float
    image_size: Optional[tuple] = None


class MonitoringResponse(BaseModel):
    """Ответ с данными мониторинга"""
    metrics: DetectionMetrics
    class_stats: List[DetectionStats]
    class_effectiveness: List[ClassEffectiveness]
    recent_detections: List[DetectionLogEntry]
    recommendations: List[str] = []
