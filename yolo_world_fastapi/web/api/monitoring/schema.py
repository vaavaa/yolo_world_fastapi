"""
Monitoring schemas.
"""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Схема ответа для проверки здоровья сервиса."""
    
    status: str
    details: dict | None = None 