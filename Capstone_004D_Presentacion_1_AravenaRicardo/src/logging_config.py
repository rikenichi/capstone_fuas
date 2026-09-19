import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """
    Formatter JSON para logs operacionales.

    No registra body, headers, query params ni datos
    introducidos por el usuario.
    """

    CAMPOS = (
        "event",
        "method",
        "path",
        "status_code",
        "duration_ms",
        "request_id",
        "error_type",
    )

    def format(self, record):
        payload = {
            "timestamp": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        for campo in self.CAMPOS:
            valor = getattr(record, campo, None)
            if valor is not None:
                payload[campo] = valor

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )


def configure_logging():
    """
    Configura únicamente el logger del proyecto.
    No modifica directamente los loggers internos de Uvicorn.
    """

    logger = logging.getLogger("capstone")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    logger.addHandler(handler)
    logger.propagate = False

    logger.info("logging_initialized")

    return logger


def add_request_logging_middleware(app):
    """
    Middleware HTTP de observabilidad.

    Registra:
    - método
    - path
    - código HTTP
    - duración
    - request_id

    No registra contenido del request.
    """

    logger = logging.getLogger("capstone.api")

    @app.middleware("http")
    async def request_logging(request, call_next):
        request_id = str(uuid.uuid4())
        inicio = time.perf_counter()

        try:
            response = await call_next(request)

            duracion_ms = round(
                (time.perf_counter() - inicio) * 1000,
                2
            )

            logger.info(
                "request_completed",
                extra={
                    "method": request.method,
                    # Solo path. No query string.
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duracion_ms,
                    "request_id": request_id,
                },
            )

            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as exc:
            duracion_ms = round(
                (time.perf_counter() - inicio) * 1000,
                2
            )

            logger.error(
                "request_failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duracion_ms,
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                },
            )

            raise
