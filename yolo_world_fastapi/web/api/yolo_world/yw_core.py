import os
import time
from typing import Tuple, Union, BinaryIO
import asyncio

import cv2
import numpy as np
import supervision as sv
from fastapi import FastAPI, UploadFile

from yolo_world_fastapi import _shared, settings
from yolo_world_fastapi._shared import clip
from yolo_world_fastapi.web.api.yolo_world.yw_utils import combine_detections, optimized_combined_filter
from yolo_world_fastapi.services.detection_monitor import detection_monitor

here = os.path.dirname(__file__)

config = settings.settings


def optimized_filter_merged_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    containment_threshold: float = 0.7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Оптимизированная фильтрация детекций, которые объединяют в себе две или более других детекций.
    
    Args:
        boxes: (N, 4) массив боксов
        scores: (N,) массив оценок
        class_ids: (N,) массив ID классов
        containment_threshold: порог для определения содержания
    
    Returns:
        tuple: (filtered_boxes, filtered_scores, filtered_class_ids)
    """
    if len(boxes) <= 2:
        return boxes, scores, class_ids
    
    keep_indices = []
    
    for i in range(len(boxes)):
        current_box = boxes[i]
        contained_count = 0
        
        # Векторизованная проверка содержания
        other_boxes = np.concatenate([boxes[:i], boxes[i+1:]])
        
        # Вычисляем пересечения
        x1_inter = np.maximum(current_box[0], other_boxes[:, 0])
        y1_inter = np.maximum(current_box[1], other_boxes[:, 1])
        x2_inter = np.minimum(current_box[2], other_boxes[:, 2])
        y2_inter = np.minimum(current_box[3], other_boxes[:, 3])
        
        # Проверяем валидность пересечений
        valid_intersection = (x2_inter > x1_inter) & (y2_inter > y1_inter)
        
        if np.any(valid_intersection):
            intersection_areas = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
            
            # Вычисляем коэффициенты содержания
            containment_ratios = np.where(
                other_areas > 0,
                intersection_areas / other_areas,
                0.0
            )
            
            # Считаем количество детекций, которые содержатся в текущей
            contained_count = np.sum(containment_ratios >= containment_threshold)
        
        # Оставляем детекцию только если она не содержит две или более других детекций
        if contained_count < 2:
            keep_indices.append(i)
    
    if len(keep_indices) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)
    
    return boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]


def filter_merged_detections(
    detections: sv.Detections, 
    containment_threshold: float = 0.7
) -> sv.Detections:
    """
    Фильтрует детекции, которые объединяют в себе две или более других детекций.
    
    Аргументы:
        detections (sv.Detections): Детекции для фильтрации.
        containment_threshold (float): Порог для определения содержания одной детекции в другой.
                                     Если детекция содержит две или более других детекций с этим порогом, она удаляется.
    
    Возвращает:
        sv.Detections: Отфильтрованные детекции без объединенных боксов.
    """
    if len(detections) <= 2:
        return detections
    
    # Используем оптимизированную функцию
    filtered_boxes, filtered_scores, filtered_class_ids = optimized_filter_merged_detections(
        detections.xyxy,
        detections.confidence,
        detections.class_id,
        containment_threshold
    )
    
    if len(filtered_boxes) == 0:
        return sv.Detections.empty()
    
    return sv.Detections(
        xyxy=filtered_boxes,
        mask=detections.mask if detections.mask is None else None,
        confidence=filtered_scores,
        class_id=filtered_class_ids,
        tracker_id=detections.tracker_id if detections.tracker_id is None else None,
    )


def non_maximum_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    score_threshold: float,
    max_num_detections: int,
    nms_inference_session,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Выполняет Non-Maximum Suppression (NMS) для отбора лучших предсказанных боксов.

    Аргументы:
        boxes (np.ndarray): Массив ограничивающих прямоугольников (N, 4).
        scores (np.ndarray): Массив оценок (N, num_classes).
        iou_threshold (float): Порог IoU для подавления перекрывающихся боксов.
        score_threshold (float): Минимальный порог оценки для отбора бокса.
        max_num_detections (int): Максимальное количество боксов после NMS.
        nms_inference_session: ONNX Runtime сессия для NMS.

    Возвращает:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Отобранные боксы (M, 4)
            - Оценки (M,)
            - Метки классов (M,)
    """
    selected_indices = nms_inference_session.run(
        output_names=["selected_indices"],
        input_feed={
            "boxes": boxes[None, :, :],
            "scores": scores[None, :, :].transpose(0, 2, 1),
            "max_output_boxes_per_class": np.array(
                [max_num_detections], dtype=np.int64
            ),
            "iou_threshold": np.array([iou_threshold], dtype=np.float32),
            "score_threshold": np.array([score_threshold], dtype=np.float32),
        },
    )[0]
    labels = selected_indices[:, 1]
    box_indices = selected_indices[:, 2]
    boxes = boxes[box_indices]
    scores = scores[box_indices, labels]

    if len(boxes) > max_num_detections:
        keep_indices = np.argsort(scores)[-max_num_detections:]
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]

    return boxes, scores, labels


async def run_yw_core(
    app: FastAPI,
    image_file: Union[str, bytes, np.ndarray, BinaryIO, UploadFile],
    class_names: list[str] = None,
    iou_threshold: float = 0.95,
    score_threshold: float = None,
    max_num_detections: int = None,
    visualize: bool = False,
    return_bytes: bool = False,
):
    """
    Основная функция инференса YOLO World.

    Аргументы:
        app (FastAPI): Экземпляр FastAPI с необходимыми сессиями в app.state.
        image_file (Union[str, bytes, np.ndarray, BinaryIO, UploadFile]):
            - Путь к изображению для инференса (str)
            - Байты изображения (bytes)
            - Массив numpy с изображением (np.ndarray)
            - Файловый объект изображения (BinaryIO)
            - Объект UploadFile из FastAPI
        class_names (list[str], optional): Список имён классов для поиска. По умолчанию берется из настроек.
        iou_threshold (float, optional): Порог IoU для NMS. По умолчанию 0.95 (95%).
        score_threshold (float, optional): Порог оценки для NMS. По умолчанию берется из настроек.
        max_num_detections (int, optional): Максимальное количество детекций. По умолчанию берется из настроек.
        visualize (bool, optional): Если True, вернуть визуализированную картинку. По умолчанию False.
        return_bytes (bool, optional): Если True, вернуть байты изображения, иначе numpy-массив. По умолчанию False.

    Возвращает:
        Если visualize=False:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Отобранные боксы (M, 4)
                - Метки классов (M,)
                - Оценки (M,)
        Если visualize=True:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray|bytes]:
                - Отобранные боксы (M, 4)
                - Метки классов (M,)
                - Оценки (M,)
                - Визуализированная картинка (numpy-массив или байты)
    """
    start_time = time.time()
    loop = asyncio.get_running_loop()

    # Обработка различных типов входных данных
    image = cv2.imdecode(np.frombuffer(image_file, np.uint8), cv2.IMREAD_COLOR)

    # Используем значения из настроек, если параметры не указаны
    if class_names is None or len(class_names) == 0 or len(class_names[0]) == 0:
        class_names = config.label_classes
    else:
        class_names = class_names[0].split(",")

    if iou_threshold is None:
        iou_threshold = config.iou_threshold

    if score_threshold is None:
        score_threshold = config.score_threshold

    if max_num_detections is None:
        max_num_detections = config.max_num_detections

    yolo_world_session = app.state.yolo_world_session
    image_size = app.state.image_size
    nms_inference_session = app.state.nms_inference_session
    textual_inference_session = app.state.textual_inference_session

    input_image, original_image_hw, padding_hw = await loop.run_in_executor(
        None, 
        _shared.transform_image, 
        image, 
        image_size,
        config.image_interpolation,
        config.image_quality_mode
    )

    tokens = clip.tokenize(class_names + [" "])
    text_feats_tuple = await loop.run_in_executor(
        None, textual_inference_session.run, None, {"input": tokens}
    )
    (text_feats,) = text_feats_tuple
    text_feats = text_feats / np.linalg.norm(text_feats, ord=2, axis=1, keepdims=True)

    scores_bboxes = await loop.run_in_executor(
        None,
        yolo_world_session.run,
        ["scores", "boxes"],
        {
            "images": input_image[None],
            "text_features": text_feats[None],
        },
    )
    scores, bboxes = scores_bboxes

    scores = scores[0]
    bboxes = bboxes[0]
    nms_result = await loop.run_in_executor(
        None,
        non_maximum_suppression,
        bboxes,
        scores,
        iou_threshold,
        score_threshold,
        max_num_detections,
        nms_inference_session,
    )
    bboxes, scores, labels = nms_result

    bboxes = await loop.run_in_executor(
        None,
        _shared.untransform_bboxes,
        bboxes,
        image_size,
        original_image_hw,
        padding_hw,
    )
    if len(bboxes) == 0:
        return {"detections": []}, image_file

    # Оптимизированная объединенная фильтрация в одном проходе
    image_area = image.shape[0] * image.shape[1]  # Высота * Ширина
    
    final_bboxes, final_scores, final_labels = optimized_combined_filter(
        boxes=bboxes,
        scores=scores,
        class_ids=labels,
        image_area=image_area,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        min_area_ratio=config.min_area_ratio,
        max_detections=max_num_detections
    )
    
    if len(final_bboxes) == 0:
        return {"detections": []}, image_file

    if visualize:
        # Визуализируем картинку с помощью _shared.visualize_bboxes
        vis_img = _shared.visualize_bboxes(
            image, final_bboxes, final_labels, final_scores, np.array(class_names),
            return_bytes=return_bytes
        )
    else:
        vis_img = image_file

    results = [
        {
            str(i): bbox.tolist(),
            "label": int(label),
            "label_name": class_names[int(label)],
            "score": float(score)
        }
        for i, (bbox, label, score) in enumerate(zip(final_bboxes, final_labels, final_scores))
    ]
    
    # Логируем результат детекции для мониторинга
    processing_time = time.time() - start_time
    detected_classes = [class_names[int(label)] for label in final_labels]
    confidences = final_scores.tolist()
    image_size = (image.shape[0], image.shape[1])  # (height, width)
    
    detection_monitor.log_detection(
        classes_requested=class_names,
        classes_detected=detected_classes,
        confidences=confidences,
        processing_time=processing_time,
        image_size=image_size
    )
    
    return {"bboxes": results}, vis_img
