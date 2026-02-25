from fastapi.routing import APIRouter

from yolo_world_fastapi.web.api import docs, echo, monitoring, yolo_world

api_router = APIRouter()
api_router.include_router(monitoring.router)
api_router.include_router(docs.router)
api_router.include_router(echo.router, prefix="/echo", tags=["echo"])
api_router.include_router(yolo_world.router, prefix="/yworld", tags=["yolo_world"])
