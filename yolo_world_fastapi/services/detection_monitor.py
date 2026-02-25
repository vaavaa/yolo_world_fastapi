import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict, deque
import logging

from yolo_world_fastapi.web.api.monitoring.detection_schema import (
    DetectionStats, ClassEffectiveness, DetectionMetrics, 
    DetectionLogEntry, MonitoringResponse
)

logger = logging.getLogger(__name__)


class DetectionMonitor:
    """Сервис для мониторинга и анализа детекций"""
    
    def __init__(self, max_log_entries: int = 1000):
        self.max_log_entries = max_log_entries
        self.detection_logs: deque = deque(maxlen=max_log_entries)
        self.class_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total_detections': 0,
            'confidence_sum': 0.0,
            'min_confidence': 1.0,
            'max_confidence': 0.0,
            'detection_count': 0,
            'request_count': 0
        })
        self.total_requests = 0
        self.total_processing_time = 0.0
        
    def log_detection(
        self,
        classes_requested: List[str],
        classes_detected: List[str],
        confidences: List[float],
        processing_time: float,
        image_size: Optional[tuple] = None
    ) -> str:
        """Логирует результат детекции"""
        request_id = str(uuid.uuid4())[:8]
        
        # Создаем запись лога
        log_entry = DetectionLogEntry(
            timestamp=datetime.now(),
            request_id=request_id,
            classes_requested=classes_requested,
            classes_detected=classes_detected,
            confidences=confidences,
            processing_time=processing_time,
            image_size=image_size
        )
        
        # Добавляем в лог
        self.detection_logs.append(log_entry)
        
        # Обновляем статистику
        self._update_stats(classes_requested, classes_detected, confidences, processing_time)
        
        logger.info(f"Detection logged: {request_id}, classes: {classes_detected}, "
                   f"confidences: {[f'{c:.3f}' for c in confidences]}, "
                   f"time: {processing_time:.3f}s")
        
        return request_id
    
    def _update_stats(
        self,
        classes_requested: List[str],
        classes_detected: List[str],
        confidences: List[float],
        processing_time: float
    ):
        """Обновляет внутреннюю статистику"""
        self.total_requests += 1
        self.total_processing_time += processing_time
        
        # Обновляем статистику для запрошенных классов
        for class_name in classes_requested:
            self.class_stats[class_name]['request_count'] += 1
        
        # Обновляем статистику для детектированных классов
        for class_name, confidence in zip(classes_detected, confidences):
            stats = self.class_stats[class_name]
            stats['total_detections'] += 1
            stats['confidence_sum'] += confidence
            stats['min_confidence'] = min(stats['min_confidence'], confidence)
            stats['max_confidence'] = max(stats['max_confidence'], confidence)
            stats['detection_count'] += 1
    
    def get_detection_stats(self) -> List[DetectionStats]:
        """Возвращает статистику по классам"""
        stats_list = []
        
        for class_name, stats in self.class_stats.items():
            if stats['total_detections'] > 0:
                avg_confidence = stats['confidence_sum'] / stats['total_detections']
            else:
                avg_confidence = 0.0
            
            stats_list.append(DetectionStats(
                class_name=class_name,
                total_detections=stats['total_detections'],
                avg_confidence=avg_confidence,
                min_confidence=stats['min_confidence'] if stats['min_confidence'] < 1.0 else 0.0,
                max_confidence=stats['max_confidence'],
                detection_count=stats['detection_count']
            ))
        
        return sorted(stats_list, key=lambda x: x.total_detections, reverse=True)
    
    def get_class_effectiveness(self) -> List[ClassEffectiveness]:
        """Возвращает анализ эффективности классов"""
        effectiveness_list = []
        
        for class_name, stats in self.class_stats.items():
            if stats['request_count'] > 0:
                detection_rate = (stats['detection_count'] / stats['request_count']) * 100
            else:
                detection_rate = 0.0
            
            if stats['total_detections'] > 0:
                avg_confidence = stats['confidence_sum'] / stats['total_detections']
            else:
                avg_confidence = 0.0
            
            effectiveness_list.append(ClassEffectiveness(
                class_name=class_name,
                detection_rate=detection_rate,
                avg_confidence=avg_confidence,
                total_requests=stats['request_count'],
                successful_detections=stats['detection_count']
            ))
        
        return sorted(effectiveness_list, key=lambda x: x.detection_rate, reverse=True)
    
    def get_metrics(self) -> DetectionMetrics:
        """Возвращает общие метрики"""
        if self.total_requests > 0:
            avg_detections = sum(stats['total_detections'] for stats in self.class_stats.values()) / self.total_requests
            avg_processing_time = self.total_processing_time / self.total_requests
        else:
            avg_detections = 0.0
            avg_processing_time = 0.0
        
        # Находим наиболее и наименее детектируемые классы
        class_stats_list = self.get_detection_stats()
        most_detected = [s.class_name for s in class_stats_list[:5]]
        least_detected = [s.class_name for s in class_stats_list[-5:] if s.total_detections > 0]
        
        return DetectionMetrics(
            total_requests=self.total_requests,
            total_detections=sum(stats['total_detections'] for stats in self.class_stats.values()),
            avg_detections_per_image=avg_detections,
            avg_processing_time=avg_processing_time,
            most_detected_classes=most_detected,
            least_detected_classes=least_detected
        )
    
    def get_recent_detections(self, limit: int = 50) -> List[DetectionLogEntry]:
        """Возвращает последние детекции"""
        return list(self.detection_logs)[-limit:]
    
    def get_recommendations(self) -> List[str]:
        """Генерирует рекомендации на основе статистики"""
        recommendations = []
        
        effectiveness = self.get_class_effectiveness()
        stats = self.get_detection_stats()
        
        # Анализируем неэффективные классы
        low_detection_classes = [
            e for e in effectiveness 
            if e.detection_rate < 5.0 and e.total_requests > 10
        ]
        
        if low_detection_classes:
            class_names = [c.class_name for c in low_detection_classes]
            recommendations.append(
                f"Классы с низкой детекцией (<5%): {', '.join(class_names)}. "
                f"Рассмотрите возможность их удаления или замены на более общие."
            )
        
        # Анализируем классы с низкой уверенностью
        low_confidence_classes = [
            s for s in stats 
            if s.avg_confidence < 0.3 and s.total_detections > 5
        ]
        
        if low_confidence_classes:
            class_names = [c.class_name for c in low_confidence_classes]
            recommendations.append(
                f"Классы с низкой уверенностью (<0.3): {', '.join(class_names)}. "
                f"Возможно, стоит увеличить score_threshold или пересмотреть эти классы."
            )
        
        # Анализируем неиспользуемые классы
        unused_classes = [
            e for e in effectiveness 
            if e.total_requests == 0
        ]
        
        if unused_classes:
            class_names = [c.class_name for c in unused_classes]
            recommendations.append(
                f"Неиспользуемые классы: {', '.join(class_names)}. "
                f"Рекомендуется их удалить для оптимизации производительности."
            )
        
        # Рекомендации по производительности
        if self.total_requests > 0:
            avg_time = self.total_processing_time / self.total_requests
            if avg_time > 2.0:
                recommendations.append(
                    f"Среднее время обработки высокое ({avg_time:.2f}s). "
                    f"Рассмотрите уменьшение количества классов или оптимизацию модели."
                )
        
        return recommendations
    
    def get_monitoring_data(self) -> MonitoringResponse:
        """Возвращает полные данные мониторинга"""
        return MonitoringResponse(
            metrics=self.get_metrics(),
            class_stats=self.get_detection_stats(),
            class_effectiveness=self.get_class_effectiveness(),
            recent_detections=self.get_recent_detections(),
            recommendations=self.get_recommendations()
        )
    
    def reset_stats(self):
        """Сбрасывает всю статистику"""
        self.detection_logs.clear()
        self.class_stats.clear()
        self.total_requests = 0
        self.total_processing_time = 0.0
        logger.info("Detection monitoring stats reset")


# Глобальный экземпляр монитора
detection_monitor = DetectionMonitor()
