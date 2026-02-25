"""Кастомные response классы для оптимизации производительности."""

import orjson
from fastapi.responses import Response
from typing import Any


class ORJSONResponse(Response):
    """Кастомный JSONResponse с использованием orjson для лучшей производительности."""
    media_type = "application/json"

    def __init__(
        self,
        content: Any,
        status_code: int = 200,
        headers: dict = None,
        media_type: str = None,
        background = None,
    ):
        super().__init__(content, status_code, headers, media_type, background)

    def render(self, content: Any) -> bytes:
        """Рендеринг контента с использованием orjson для максимальной производительности."""
        return orjson.dumps(
            content,
            option=orjson.OPT_SERIALIZE_NUMPY |  # Сериализация numpy массивов
                   orjson.OPT_OMIT_MICROSECONDS |  # Пропуск микросекунд в datetime
                   orjson.OPT_NON_STR_KEYS  # Поддержка не-строковых ключей
        )
