# Capstone FUAS

## Seguridad y privacidad

La aplicación aplica principios de minimización de datos y validación de entradas.

### Datos aceptados por la API

La inferencia utiliza exclusivamente las siguientes variables:

- `NEM`
- `QUINTIL_SE4`
- `COD_DEPE`
- `EDAD`
- `GENERO`
- `NACIONALIDAD`

El esquema de entrada está configurado con `extra="forbid"`, por lo que cualquier campo adicional es rechazado automáticamente.

### Identificadores personales

La API no requiere ni admite identificadores personales como:

- RUT
- RUN
- MRUN

Estos campos no forman parte de las propiedades permitidas por el esquema OpenAPI.

El identificador `MRUN`, utilizado durante el procesamiento histórico de los datos para realizar joins entre fuentes oficiales, no forma parte de los datos analíticos utilizados por el modelo ni de la interfaz de predicción.

### Validación de entradas

Antes de ejecutar el modelo se validan los rangos y categorías admitidos:

- NEM: 100 a 700.
- Quintil socioeconómico: 1 a 5.
- Dependencia del establecimiento: códigos 1 a 6.
- Edad: 10 a 130 años.
- Género: categorías 1 y 2.
- Nacionalidad: categorías 0, 1 y 2.

La validación se realiza tanto en el esquema Pydantic de la API como en la capa de inferencia.

### Minimización de datos

Las respuestas de `/predict` no devuelven nuevamente las variables proporcionadas por el usuario.

La aplicación no implementa almacenamiento de consultas individuales ni persistencia de perfiles ingresados en la interfaz.

### Manejo de errores

Los errores de validación generan respuestas HTTP `422`.

Los errores internos generan respuestas HTTP `500` con mensajes genéricos, evitando exponer:

- rutas internas del servidor;
- stack traces;
- nombres de archivos internos;
- detalles de implementación sensibles.

### Logging

Los logs estándar de Uvicorn registran información operacional como:

- método HTTP;
- endpoint solicitado;
- código de respuesta.

No se registra el contenido JSON de los formularios de predicción.

### Arquitectura web

El frontend React compilado es servido directamente por FastAPI desde el mismo origen que la API.

Por esta razón, la versión actual del MVP no requiere una política CORS abierta entre frontend y backend.

### Interpretación del resultado

La salida del modelo corresponde a una estimación orientativa basada en patrones históricos.

No constituye una decisión oficial de asignación de gratuidad, becas, créditos u otros beneficios estudiantiles.


## Despliegue

La aplicación se encuentra desplegada públicamente en Render:

https://capstone-fuas.onrender.com

El servicio integra en una única aplicación:

- Frontend React compilado.
- API FastAPI.
- Modelo Random Forest congelado.
- Explicaciones SHAP.
- Endpoints de salud y metadatos del modelo.

### Endpoints principales

- `/` — Interfaz web.
- `/health` — Estado del servicio y disponibilidad del modelo.
- `/model-info` — Información del modelo y métricas.
- `/predict` — Predicción individual.
- `/explain` — Explicación SHAP de la estimación.

El modelo fue desarrollado con datos FUAS 2024 y evaluado mediante holdout temporal 2025.

## Dashboard histórico

La aplicación incluye una vista de análisis histórico FUAS para el período 2008–2025.

El dashboard presenta:

- evolución anual de postulantes;
- cantidad de beneficiarios;
- tasa histórica de beneficio;
- análisis por género;
- análisis por quintil socioeconómico;
- análisis por tramos de edad;
- cobertura metodológica de las variables según año.

La disponibilidad de variables cambia según el período histórico. Por ejemplo, `QUINTIL_SE4` no está disponible en los archivos FUAS 2019–2021, por lo que esos años no se interpretan como valor cero.

La tasa de beneficio utiliza la misma definición de beneficio válido empleada en el pipeline del proyecto. Los casos cuyo único registro corresponde a una sigla no resuelta se excluyen del cálculo de la tasa.

### Endpoint del dashboard

- `/dashboard` — Datos agregados históricos utilizados por la interfaz.

El dashboard es descriptivo y no modifica ni reentrena el modelo predictivo. El modelo final continúa utilizando 2024 para desarrollo y 2025 como evaluación temporal.
