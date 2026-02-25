import base64
from typing import Tuple, Dict, Any
import orjson

from fastapi import APIRouter, Depends, Request
from starlette.responses import Response

from yolo_world_fastapi.web.api.yolo_world.schema import (YoloWorldInputDTO,
                                                          YoloWorldOnlyBoxes)
from yolo_world_fastapi.web.api.yolo_world.yw_core import run_yw_core
from yolo_world_fastapi.web.responses import ORJSONResponse

router = APIRouter()


async def process_image(
    img_stream_dto: YoloWorldInputDTO,
    request: Request) -> Tuple[dict, bytes]:
    file_content = await img_stream_dto.img_stream.read()
    bdict, vis_img_bytes = await run_yw_core(
        request.app,
        file_content,
        class_names=img_stream_dto.class_names if hasattr(img_stream_dto,
                                                              'class_names') else None,
        iou_threshold=img_stream_dto.iou_threshold if hasattr(img_stream_dto,
                                                              'iou_threshold') else None,
        score_threshold=img_stream_dto.score_threshold if hasattr(img_stream_dto,
                                                                  'score_threshold') else None,
        max_num_detections=img_stream_dto.max_num_detections if hasattr(img_stream_dto,
                                                                        'max_num_detections') else None,
        visualize=True,
        return_bytes=True
    )

    return bdict, vis_img_bytes


@router.post("/pre-stream")
async def process_image_to_pre_stream(
    request: Request,
    img_stream_dto: YoloWorldInputDTO = Depends(),
) -> Response:
    _, vis_img_bytes = await process_image(img_stream_dto, request)
    return Response(
        content=vis_img_bytes,
        media_type="image/png",
        headers={"Content-Type": "image/png"}
    )


@router.post("/base64")
async def process_image_to_base64(
    request: Request,
    img_stream_dto: YoloWorldOnlyBoxes = Depends()
) -> ORJSONResponse:
    bdict, vis_img_bytes = await process_image(img_stream_dto, request)
    if not img_stream_dto.only_bboxs:
        bdict['img_base64'] = base64.b64encode(vis_img_bytes).decode('utf-8')
        return ORJSONResponse(content=bdict)
    
    return ORJSONResponse(content=bdict)
