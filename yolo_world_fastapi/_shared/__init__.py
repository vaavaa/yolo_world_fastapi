import os
from typing import List
from typing import Tuple

import cv2
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))


def get_interpolation_method(method: str) -> int:
    """
    Возвращает константу OpenCV для метода интерполяции.
    
    Args:
        method: Название метода ("linear", "cubic", "lanczos", "area")
    
    Returns:
        Константа OpenCV для интерполяции
    """
    interpolation_map = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
    }
    return interpolation_map.get(method.lower(), cv2.INTER_CUBIC)


def get_optimal_interpolation(original_size: Tuple[int, int], target_size: int) -> int:
    """
    Выбирает оптимальный метод интерполяции в зависимости от изменения размера.
    
    Args:
        original_size: Исходный размер (height, width)
        target_size: Целевой размер
    
    Returns:
        Константа OpenCV для интерполяции
    """
    original_max = max(original_size)
    scale_factor = target_size / original_max
    
    if scale_factor > 1.5:  # Увеличение изображения
        return cv2.INTER_CUBIC  # Лучше для увеличения
    elif scale_factor < 0.5:  # Значительное уменьшение
        return cv2.INTER_AREA   # Лучше для уменьшения
    else:  # Небольшие изменения
        return cv2.INTER_CUBIC  # Хороший баланс


def transform_image(
    image: np.ndarray, 
    image_size: int,
    interpolation_method: str = "cubic",
    quality_mode: str = "balanced"
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Преобразует изображение для YOLO World с улучшенным качеством.
    
    Args:
        image: Исходное изображение
        image_size: Целевой размер (обычно 640)
        interpolation_method: Метод интерполяции ("linear", "cubic", "lanczos", "area")
        quality_mode: Режим качества ("fast", "balanced", "high_quality")
    
    Returns:
        Tuple с преобразованным изображением, исходными размерами и padding
    """
    height, width = image.shape[:2]

    # Выбираем метод интерполяции
    if quality_mode == "fast":
        interpolation = cv2.INTER_LINEAR
    elif quality_mode == "high_quality":
        interpolation = cv2.INTER_LANCZOS4
    elif quality_mode == "balanced":
        # Автоматический выбор оптимального метода
        interpolation = get_optimal_interpolation((height, width), image_size)
    else:
        # Используем указанный метод
        interpolation = get_interpolation_method(interpolation_method)

    scale = image_size / max(height, width)
    
    # Применяем улучшенное масштабирование
    image_resized = cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=interpolation,
    )
    pad_height = image_size - image_resized.shape[0]
    pad_width = image_size - image_resized.shape[1]
    image_resized = np.pad(
        image_resized,
        (
            (pad_height // 2, pad_height - pad_height // 2),
            (pad_width // 2, pad_width - pad_width // 2),
            (0, 0),
        ),
        mode="constant",
        constant_values=114,
    )
    input_image = image_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return input_image, (height, width), (pad_height, pad_width)


def untransform_bboxes(
    bboxes: np.ndarray,
    image_size: int,
    original_image_hw: Tuple[int, int],
    padding_hw: Tuple[int, int],
) -> np.ndarray:
    bboxes -= np.array([padding_hw[1] // 2, padding_hw[0] // 2] * 2)
    bboxes /= image_size / max(original_image_hw)
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, original_image_hw[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, original_image_hw[0])
    bboxes = bboxes.round().astype(int)
    return bboxes


def visualize_bboxes(
    image: np.ndarray,
    bboxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: np.ndarray,
    return_bytes: bool = False,
) -> np.ndarray | bytes:
    image_viz = image.copy()
    for bbox, label, score in zip(bboxes, labels, scores):
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0)
        cv2.rectangle(image_viz, (x1, y1), (x2, y2), color, 2)
        caption = f"{class_names[label]}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, image.shape[0] / 640.0)
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(caption, font, font_scale, thickness)
        cv2.rectangle(image_viz, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
        cv2.putText(image_viz, caption, (x1, y1 - baseline), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    if return_bytes:
        success, buffer = cv2.imencode('.png', image_viz)
        if not success:
            raise RuntimeError("Не удалось закодировать изображение в PNG")
        return buffer.tobytes()
    else:
        return image_viz

