from fastapi import UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional


class YoloWorldInputDTO(BaseModel):
    img_stream: UploadFile = File(...)
    class_names: Optional[List[str]] = None
    iou_threshold: Optional[float] = 0.5
    score_threshold: Optional[float] = 0.20
    max_num_detections: Optional[int] = 5

class YoloWorldOnlyBoxes(YoloWorldInputDTO):
    only_bboxs: Optional[bool] = False
