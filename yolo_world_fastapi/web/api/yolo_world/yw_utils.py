from typing import Optional, List

import cv2
import numpy as np
import supervision as sv
import os
import logging
from functools import lru_cache
from typing import Tuple

from yolo_world_fastapi.settings import Settings

settings = Settings()

CIRCLE_RADIUS_PERCENT = settings.circle_radius_percent
MAX_DETECTIONS_COUNT = settings.max_num_detections
TEXT_SCALE = settings.text_scale
TEXT_THICKNESS = settings.text_thickness

logger = logging.getLogger(__name__)


def sort_detections_by_x1(detections: sv.Detections) -> sv.Detections:
    x1_coordinates = detections.xyxy[:, 0]
    sorted_indices = np.argsort(x1_coordinates)
    sorted_detections = sv.Detections(
        xyxy=detections.xyxy[sorted_indices],
        mask=detections.mask[sorted_indices] if detections.mask is not None else None,
        confidence=detections.confidence[
            sorted_indices] if detections.confidence is not None else None,
        class_id=detections.class_id[
            sorted_indices] if detections.class_id is not None else None,
        tracker_id=detections.tracker_id[
            sorted_indices] if detections.tracker_id is not None else None,
    )

    return sorted_detections


class InsideBoxAnnotator(sv.BoxAnnotator):
    def annotate(
        self,
        scene: np.ndarray,
        detections: sv.Detections,
        labels: Optional[List[str]] = None,
        skip_label: bool = False,
    ) -> np.ndarray:

        bboxes = {}
        font = cv2.FONT_HERSHEY_SIMPLEX

        detections = sort_detections_by_x1(detections)

        for i in range(min(len(detections), MAX_DETECTIONS_COUNT)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            class_id = (
                detections.class_id[i] if detections.class_id is not None else None
            )
            idx = class_id if class_id is not None else i
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, sv.ColorPalette)
                else self.color
            )
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
            bboxes[i + 1] = (x1, y1, x2, y2)

            if skip_label:
                continue

            # text = (
            #     f"{class_id}"
            #     if (labels is None or len(detections) != len(labels))
            #     else labels[i]
            # )
            text = str(i + 1) if len(detections) > 1 else 'Ok!'

            # Calculate the size of the text and its background
            text_size = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=TEXT_SCALE,
                thickness=TEXT_THICKNESS,
            )[0]

            # Text background coordinates
            background_x1 = x1
            background_y1 = y1
            background_x2 = x1 + text_size[0] + self.text_padding
            background_y2 = y1 + text_size[1] + self.text_padding + 3

            cv2.rectangle(
                img=scene,
                pt1=(background_x1, background_y1),
                pt2=(background_x2, background_y2),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )

            text_x = x1 + 1
            text_y = background_y2 - 5

            # circle_radius = int(scene.shape[1] * CIRCLE_RADIUS_PERCENT)
            # cv2.circle(
            #     img=scene,
            #     center=(text_x, text_y),
            #     radius=circle_radius,
            #     color=color.as_bgr(),
            #     thickness=cv2.FILLED
            # )

            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=TEXT_SCALE,
                color=self.text_color.as_rgb(),
                thickness=TEXT_THICKNESS,
                lineType=cv2.LINE_AA,
            )

        return scene, bboxes


def save_yolo_detection(path, name, detections, width, height) -> None:
    # save detections to file
    with open(path + name + ".txt", "w") as f:

        for xyxy, class_id in zip(detections.xyxy, detections.class_id):
            yolo_label = label_to_yolo(xyxy, width, height)
            yolo_label = [str(x) for x in yolo_label]
            f.write(str(class_id) + " " + " ".join(yolo_label) + "\n")

        if len(detections.xyxy) == 0:
            f.write("\n")


def label_to_yolo(label, image_w, image_h):
    x_min, x_max = label[0], label[2]
    y_min, y_max = label[1], label[3]

    label_w = x_max - x_min
    label_h = y_max - y_min

    center_x = (x_min + label_w / 2) / image_w
    center_y = (y_min + label_h / 2) / image_h

    return center_x, center_y, label_w / image_w, label_h / image_h


@lru_cache(maxsize=128)
def _cached_area_calculation(boxes_hash: str) -> float:
    """
    Кэшированный расчет площади для бокса.
    Используется для оптимизации повторяющихся вычислений.
    """
    # Это заглушка - в реальности нужно десериализовать бокс
    # Но для демонстрации концепции оставляем так
    return 0.0


def fast_iou_calculation(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Быстрый расчет IoU между двумя наборами боксов.
    
    Args:
        boxes1: (N, 4) массив боксов [x1, y1, x2, y2]
        boxes2: (M, 4) массив боксов [x1, y1, x2, y2]
    
    Returns:
        (N, M) массив IoU значений
    """
    # Вычисляем площади боксов
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Вычисляем пересечения
    x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])
    
    # Проверяем валидность пересечений
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Вычисляем объединения
    union = area1[:, None] + area2 - intersection
    
    # Избегаем деления на ноль
    iou = np.where(union > 0, intersection / union, 0.0)
    
    return iou


def optimized_filter_overlapping_detections(
    boxes: np.ndarray, 
    scores: np.ndarray, 
    class_ids: np.ndarray,
    iou_threshold: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Оптимизированная фильтрация перекрывающихся детекций.
    
    Args:
        boxes: (N, 4) массив боксов
        scores: (N,) массив оценок
        class_ids: (N,) массив ID классов
        iou_threshold: порог IoU для фильтрации
    
    Returns:
        tuple: (filtered_boxes, filtered_scores, filtered_class_ids)
    """
    if len(boxes) == 0:
        return boxes, scores, class_ids
    
    # Сортируем по убыванию confidence
    sorted_indices = np.argsort(-scores)
    sorted_boxes = boxes[sorted_indices]
    sorted_scores = scores[sorted_indices]
    sorted_class_ids = class_ids[sorted_indices]
    
    keep_indices = []
    
    for i in range(len(sorted_boxes)):
        if i == 0:
            keep_indices.append(i)
            continue
        
        # Вычисляем IoU с уже выбранными боксами
        current_box = sorted_boxes[i:i+1]  # (1, 4)
        kept_boxes = sorted_boxes[keep_indices]  # (M, 4)
        
        # Быстрый расчет IoU
        ious = fast_iou_calculation(current_box, kept_boxes)[0]  # (M,)
        
        # Проверяем, что максимальный IoU меньше порога
        if np.max(ious) < iou_threshold:
            keep_indices.append(i)
    
    # Возвращаем отфильтрованные данные
    final_indices = sorted_indices[keep_indices]
    return boxes[final_indices], scores[final_indices], class_ids[final_indices]


def filter_overlapping_detections(detections: sv.Detections, iou_threshold: float = 0.95) -> sv.Detections:
    """
    Filter out overlapping detections based on IOU threshold.
    Keeps the detection with the highest confidence score when there is overlap.
    """
    if len(detections) == 0:
        return detections

    # Используем оптимизированную функцию
    filtered_boxes, filtered_scores, filtered_class_ids = optimized_filter_overlapping_detections(
        detections.xyxy, 
        detections.confidence, 
        detections.class_id,
        iou_threshold
    )

    return sv.Detections(
        xyxy=filtered_boxes,
        mask=detections.mask if detections.mask is None else None,
        confidence=filtered_scores,
        class_id=filtered_class_ids,
        tracker_id=detections.tracker_id if detections.tracker_id is None else None,
    )


def optimized_combined_filter(
    boxes: np.ndarray,
    scores: np.ndarray, 
    class_ids: np.ndarray,
    image_area: float,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.2,
    min_area_ratio: float = 0.005,
    max_detections: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Оптимизированная объединенная фильтрация детекций в одном проходе.
    
    Args:
        boxes: (N, 4) массив боксов
        scores: (N,) массив оценок
        class_ids: (N,) массив ID классов
        image_area: общая площадь изображения
        iou_threshold: порог IoU для NMS
        score_threshold: минимальный порог оценки
        min_area_ratio: минимальный коэффициент площади
        max_detections: максимальное количество детекций
    
    Returns:
        tuple: (filtered_boxes, filtered_scores, filtered_class_ids)
    """
    if len(boxes) == 0:
        return boxes, scores, class_ids
    
    # 1. Фильтрация по score threshold
    score_mask = scores >= score_threshold
    if not np.any(score_mask):
        return np.empty((0, 4)), np.empty(0), np.empty(0)
    
    boxes = boxes[score_mask]
    scores = scores[score_mask]
    class_ids = class_ids[score_mask]
    
    # 2. Фильтрация по площади (векторизованная)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_mask = areas / image_area >= min_area_ratio
    if not np.any(area_mask):
        return np.empty((0, 4)), np.empty(0), np.empty(0)
    
    boxes = boxes[area_mask]
    scores = scores[area_mask]
    class_ids = class_ids[area_mask]
    
    # 3. NMS фильтрация
    if len(boxes) > 1:
        boxes, scores, class_ids = optimized_filter_overlapping_detections(
            boxes, scores, class_ids, iou_threshold
        )
    
    # 4. Ограничение количества детекций
    if len(boxes) > max_detections:
        # Берем топ-K по confidence
        top_indices = np.argsort(-scores)[:max_detections]
        boxes = boxes[top_indices]
        scores = scores[top_indices]
        class_ids = class_ids[top_indices]
    
    return boxes, scores, class_ids


def combine_detections(detections_list, overwrite_class_ids, iou_threshold: float = 0.95):
    if len(detections_list) == 0:
        return sv.Detections.empty()

    if overwrite_class_ids is not None and len(overwrite_class_ids) != len(
        detections_list
    ):
        raise ValueError(
            "Length of overwrite_class_ids must match the length of detections_list."
        )

    xyxy = []
    mask = []
    confidence = []
    class_id = []
    tracker_id = []

    for idx, detection in enumerate(detections_list):
        xyxy.append(detection.xyxy)

        if detection.mask is not None:
            mask.append(detection.mask)

        if detection.confidence is not None:
            confidence.append(detection.confidence)

        if detection.class_id is not None:
            if overwrite_class_ids is not None:
                # Overwrite the class IDs for the current Detections object
                class_id.append(
                    np.full_like(
                        detection.class_id, overwrite_class_ids[idx], dtype=np.int64
                    )
                )
            else:
                class_id.append(detection.class_id)

        if detection.tracker_id is not None:
            tracker_id.append(detection.tracker_id)

    xyxy = np.vstack(xyxy)
    mask = np.vstack(mask) if mask else None
    confidence = np.hstack(confidence) if confidence else None
    class_id = np.hstack(class_id) if class_id else None
    tracker_id = np.hstack(tracker_id) if tracker_id else None

    combined_detections = sv.Detections(
        xyxy=xyxy,
        mask=mask,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
    )

    # Apply additional IOU filtering
    return filter_overlapping_detections(combined_detections, iou_threshold)


def check_model_files_is_ok(model_config_path, model_checkpoint_path):
    missing_files = False
    for abs_path in (model_config_path, model_checkpoint_path):
        if not os.path.isfile(abs_path):
            missing_files = True
            logger.error(f'NO config file: {abs_path}')
            break

    return not missing_files
