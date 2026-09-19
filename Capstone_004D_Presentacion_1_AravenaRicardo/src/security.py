import os

from fastapi.middleware.cors import CORSMiddleware


DEFAULT_ORIGINS = [
    "https://capstone-fuas.onrender.com",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def get_allowed_origins():
    """
    Lee ALLOWED_ORIGINS desde variables de entorno.

    Formato:
    https://dominio1.cl,https://dominio2.cl

    Si no existe, usa una lista segura para producción
    y desarrollo local.
    """
    raw = os.getenv("ALLOWED_ORIGINS")

    if not raw:
        return DEFAULT_ORIGINS

    return [
        origin.strip()
        for origin in raw.split(",")
        if origin.strip()
    ]


def configure_cors(app):
    """
    Configura CORS de forma explícita.

    No utiliza allow_origins=["*"].
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=False,
        allow_methods=[
            "GET",
            "POST",
            "OPTIONS",
        ],
        allow_headers=[
            "Content-Type",
            "Accept",
        ],
    )


def add_security_headers_middleware(app):
    """
    Agrega headers HTTP básicos de seguridad.

    No utiliza una CSP restrictiva todavía para evitar
    afectar el frontend React desplegado.
    """

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )

        return response
