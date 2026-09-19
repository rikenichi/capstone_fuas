import logging
import os
import time
from collections import defaultdict, deque

from fastapi.responses import JSONResponse


logger = logging.getLogger("capstone.rate_limit")


DEFAULT_LIMITS = {
    "/predict": 30,
    "/explain": 20,
}

WINDOW_SECONDS = 60


class InMemoryRateLimiter:
    """
    Rate limiter simple por ventana deslizante.

    Adecuado para una instancia única.
    En despliegues con múltiples instancias se debería
    utilizar un almacén compartido como Redis.
    """

    def __init__(self, limits=None, window_seconds=60):
        self.limits = limits or DEFAULT_LIMITS
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)

    def is_allowed(self, client_id, path, now=None):
        if path not in self.limits:
            return True, None

        now = now if now is not None else time.monotonic()

        key = (client_id, path)
        history = self.requests[key]

        cutoff = now - self.window_seconds

        while history and history[0] <= cutoff:
            history.popleft()

        limit = self.limits[path]

        if len(history) >= limit:
            retry_after = max(
                1,
                int(self.window_seconds - (now - history[0]))
            )
            return False, retry_after

        history.append(now)

        return True, None


def _get_client_id(request):
    """
    Obtiene una clave técnica para aplicar el límite.

    En Render se prioriza X-Forwarded-For, tomando
    únicamente la primera dirección.
    """

    forwarded_for = request.headers.get("x-forwarded-for")

    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"


def add_rate_limit_middleware(app):
    predict_limit = int(
        os.getenv("RATE_LIMIT_PREDICT", "30")
    )

    explain_limit = int(
        os.getenv("RATE_LIMIT_EXPLAIN", "20")
    )

    limiter = InMemoryRateLimiter(
        limits={
            "/predict": predict_limit,
            "/explain": explain_limit,
        },
        window_seconds=WINDOW_SECONDS,
    )

    @app.middleware("http")
    async def rate_limit_middleware(request, call_next):
        path = request.url.path

        # Solo protegemos endpoints definidos.
        if path not in limiter.limits:
            return await call_next(request)

        client_id = _get_client_id(request)

        allowed, retry_after = limiter.is_allowed(
            client_id,
            path,
        )

        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                extra={
                    "method": request.method,
                    "path": path,
                    "status_code": 429,
                },
            )

            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        "Demasiadas solicitudes. "
                        "Intente nuevamente en unos segundos."
                    )
                },
                headers={
                    "Retry-After": str(retry_after),
                },
            )

        response = await call_next(request)

        return response
