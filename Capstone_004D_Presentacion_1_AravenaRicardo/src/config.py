"""
config.py — Configuración central del pipeline ETL del Capstone FUAS.

Toda ruta, catálogo y lista de variables vive aquí para que el resto del
código trabaje por nombre y no por posición, y para que cualquier cambio de
supuesto sea explícito y auditable.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Rutas (capas de datos)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_STAGING = ROOT / "data" / "staging"
DATA_ANALYTIC = ROOT / "data" / "analytic"
REPORTS_DIR = ROOT / "artifacts" / "reports"
FIGURES_DIR = ROOT / "artifacts" / "figures"

for _d in (DATA_STAGING, DATA_ANALYTIC, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Archivos fuente por año
# ----------------------------------------------------------------------------
FUAS_FILES = {
    2024: DATA_RAW / "POSTULACIONES_FUAS_2024_WEB.csv",
    2025: DATA_RAW / "POSTULACIONES_FUAS_2025_WEB.csv",
}
ASIG_FILES = {
    2024: DATA_RAW / "Asignacion_2024_PA_PUBL.csv",
    2025: DATA_RAW / "Asignacion_2025_PA_PUBL.csv",
}

CSV_SEP = ";"

# Orden de encodings a probar (lectura robusta, no fijar latin-1 por defecto)
ENCODINGS_TRY = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

# Palabras conocidas para detectar mojibake tras leer con un encoding dado.
# Si el texto correcto NO aparece o aparecen secuencias corruptas, se descarta.
MOJIBAKE_CANARIES = ["Región", "Ñuble", "Ñuñoa", "Biobío", "Los Ríos", "Araucanía"]
MOJIBAKE_BAD_SIGNS = ["Ã", "Ã±", "Ã³", "Ã­", "Ã©", "RegiÃ³n", " Â"]

# ----------------------------------------------------------------------------
# Normalización de nombres de columnas (trabajar por nombre, nunca por posición)
# ----------------------------------------------------------------------------
# Diferencias reales detectadas entre 2024 y 2025:
#   - 2024 usa FEC_NAC_ALU; 2025 usa FEC_NAC  -> unificar a FEC_NAC
COLUMN_RENAME = {
    "FEC_NAC_ALU": "FEC_NAC",
}

# ----------------------------------------------------------------------------
# Población del modelo
# ----------------------------------------------------------------------------
# El dataset ML principal usa solo PROCESO == "FUAS". Las categorías
# "BECA VOCACION DE PROFESOR" y "BECAS DE REPARACION" se conservan aparte para
# análisis descriptivo. Justificación: mejora la homogeneidad de la cohorte y
# reduce el riesgo de que categorías administrativas especiales dominen la
# predicción (NO se afirma que elimine por completo la fuga).
PROCESO_MODELO = "FUAS"

# ----------------------------------------------------------------------------
# Target y catálogo de beneficios
# ----------------------------------------------------------------------------
# Catálogo de siglas reconocidas por MINEDUC (Anexo I de Asignaciones).
# Un valor cuenta como beneficio válido solo si, tras strip()+upper(),
# pertenece a este catálogo. Nulos, vacíos o valores desconocidos NO cuentan.
CATALOGO_BENEFICIOS = {
    "BANV", "BARCIS", "BART", "BBIC", "BEA", "BET", "BIBERO", "BHPE",
    "BJGM", "BJGME", "BNM", "BPED", "BNA", "BNC", "BNM I", "BNM II",
    "BNM III", "BPACIFICO", "BPSU", "BUDM", "BCE", "BVP", "BVP2",
    "BVP3",  # Beca Vocación de Profesor 3: no figura en el Anexo I del
             # diccionario, pero pertenece claramente a la familia BVP/BVP2
             # (Vocación de Profesor). Se agrega como beneficio válido.
    "CONTINUIDAD", "FSCU", "GRATUIDAD", "RETTIG", "REUBICACION",
    "TITULAR", "TRASPASO",
}

# Siglas desconocidas YA OBSERVADAS en los datos (no documentadas en el Anexo I).
# Sirven solo para distinguir en el reporte entre "desconocidas ya vistas" y
# "siglas nuevas" que aparezcan a futuro. La regla de tratamiento es general:
# CUALQUIER sigla no vacía y no catalogada se trata como desconocida (target NA),
# esté o no en esta lista.
BENEFICIOS_DESCONOCIDOS_EN_REVISION = {"BSA", "BPDT"}

# Clasificación conceptual de beneficios para las flags derivadas.
BENEFICIO_GRATUIDAD = {"GRATUIDAD"}
BENEFICIO_FSCU = {"FSCU"}

# Catálogo explícito de BECAS (no inferir por la primera letra: hay becas cuya
# sigla no empieza con B — RETTIG, REUBICACION, TITULAR, TRASPASO, CONTINUIDAD).
BENEFICIOS_BECA = {
    "BANV", "BARCIS", "BART", "BBIC", "BEA", "BET", "BIBERO", "BHPE",
    "BJGM", "BJGME", "BNM", "BPED", "BNA", "BNC", "BNM I", "BNM II",
    "BNM III", "BPACIFICO", "BPSU", "BUDM", "BCE", "BVP", "BVP2", "BVP3",
    "CONTINUIDAD", "RETTIG", "REUBICACION", "TITULAR", "TRASPASO",
}

# "OTRO" queda para beneficios reconocidos que no sean gratuidad, FSCU ni beca.
# Se deriva del catálogo para no mantener listas paralelas. Con el catálogo
# actual queda vacío, pero absorbe automáticamente cualquier instrumento futuro
# que no sea beca/gratuidad/FSCU.
BENEFICIO_OTRO_EXPLICITO = (
    CATALOGO_BENEFICIOS - BENEFICIOS_BECA - BENEFICIO_GRATUIDAD - BENEFICIO_FSCU
)

# ----------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------
# Clave técnica de cruce (se conserva en staging, se elimina en analytic).
KEY = "MRUN"

# Predictores del modelo base (PROCESO queda fuera porque filtramos por él).
PREDICTORS = ["NEM", "QUINTIL_SE4", "COD_DEPE", "EDAD", "GENERO", "NACIONALIDAD", "ETNIA"]

# Variables categóricas / numéricas para casteo explícito.
CAST_NUMERIC = ["EDAD", "NEM", "QUINTIL_SE4", "DECIL_SE4", "COD_DEPE",
                "GENERO", "NACIONALIDAD", "ETNIA"]

# Columnas de Asignaciones que NUNCA deben usarse como predictores (leakage:
# pertenecen al resultado o al proceso de asignación).
LEAKAGE_COLS = ["BENEFICIO_BECA_FSCU", "TIPO_ALUMNO", "QUINTIL_INGRESO", "DECIL_DFE"]

# Columnas conservadas solo para análisis (no features del modelo).
ANALYSIS_ONLY = ["ANIO_PROCESO", "TIPO_BENEFICIO", "BENEFICIOS_VALIDOS",
                 "BENEFICIOS_ASIGNADOS_RAW",
                 "TIENE_GRATUIDAD", "TIENE_BECA", "TIENE_FSCU", "TIENE_OTRO",
                 "NOMBRE_REGION", "NOMBRE_COMUNA"]

TARGET = "OBTUVO_BENEFICIO"

# ----------------------------------------------------------------------------
# ETNIA — estado de validación documental (Parte A)
# ----------------------------------------------------------------------------
# El diccionario FUAS (Anexo III) define 0 = "No pertenece a ninguna etnia" y
# 1 = "Aymara". Pero en los datos 2024 el valor 0 NO aparece, el 1 concentra
# ~85% y el rango observado es 1..12 (sin 0, 13 ni 99). La interpretación
# literal es implausible. La búsqueda en fuentes oficiales no arrojó un
# diccionario alternativo que confirme la codificación real del CSV 2024.
# Por lo tanto: ETNIA se trata como CÓDIGO NOMINAL sin interpretación semántica.
# NO se usan nombres (Aymara/Mapuche/etc.) en gráficos ni reportes de modelado.
ETNIA_CONFIRMADA = False           # cambiar a True solo con evidencia documental
ETNIA_COL = "ETNIA"                # se usa como código nominal (ETNIA_CODIGO)

# ----------------------------------------------------------------------------
# Reglas de valores implausibles (Parte B) — configurables, solo se aplican a
# TRAIN 2024. NO se aplican por defecto: el EDA no elimina nada. Un pipeline de
# modelado puede activarlas explícitamente.
# ----------------------------------------------------------------------------
# EDAD: la población objetivo son postulantes a educación superior. Edades muy
# bajas (<15) o muy altas son implausibles/errores probables, distintas de un
# simple outlier IQR (un postulante de 45 años es plausible).
EDAD_MIN_PLAUSIBLE = 15
EDAD_MAX_PLAUSIBLE = 90            # sobre 90 se considera implausible/error
# NEM: rango operativo plausible DEFINIDO PARA ESTE PROYECTO (no rango oficial).
# El dataset contiene valores inferiores; 100 se trata como centinela/error y
# <300 como rango a revisar. Ajustable si se confirma una fuente oficial.
NEM_MIN_PLAUSIBLE = 300
NEM_MAX_PLAUSIBLE = 700
NEM_CENTINELA = {100}             # valores tratados como centinela (no reales)

# Política de aplicación: "flag" (marca sin eliminar), "drop_train" (elimina
# solo en train), "clip" (recorta a los límites). Por defecto: flag.
REGLA_VALORES_IMPLAUSIBLES = "flag"

# ----------------------------------------------------------------------------
# Categorías raras (Parte E) — regla reproducible, parametrizada. NO se decide
# con 2025.
# ----------------------------------------------------------------------------
RARE_THRESHOLD = 0.01             # < 1% del total = categoría rara

# ----------------------------------------------------------------------------
# Split interno de 2024 (Parte G) — 2025 permanece aislado.
# ----------------------------------------------------------------------------
RANDOM_SEED = 42
VALID_SIZE = 0.20                 # 80/20 train/validation estratificado

# ----------------------------------------------------------------------------
# Roles de variables para el pipeline (Parte C/D)
# ----------------------------------------------------------------------------
FEATURES_NUMERIC = ["NEM", "EDAD"]
FEATURES_ORDINAL = ["QUINTIL_SE4"]              # orden natural 1..5, no one-hot
FEATURES_NOMINAL = ["COD_DEPE", "GENERO", "NACIONALIDAD", "ETNIA"]

# Nominales autorizadas para agrupación de categorías raras. ETNIA queda FUERA
# mientras su codificación no esté confirmada (ETNIA_CONFIRMADA = False): no se
# agrupa una variable cuya semántica aún no podemos defender. Si en el futuro se
# confirma, se puede añadir "ETNIA" a esta lista.
FEATURES_NOMINAL_AGRUPABLES = ["COD_DEPE", "GENERO", "NACIONALIDAD"]

# Variables prohibidas como features (identificadores, metadatos o derivadas
# del resultado). El pipeline debe verificar que ninguna entre.
FORBIDDEN_FEATURES = [
    "MRUN", "ANIO_PROCESO", "TIPO_BENEFICIO", "BENEFICIOS_VALIDOS",
    "BENEFICIOS_ASIGNADOS_RAW", "TIENE_GRATUIDAD", "TIENE_BECA",
    "TIENE_FSCU", "TIENE_OTRO", "NOMBRE_REGION", "NOMBRE_COMUNA",
    "OBTUVO_BENEFICIO_RAW", "SOLO_DESCONOCIDO", "TIENE_VALIDO",
]

# Agrupación documentada de COD_DEPE (alternativa B, para comparar).
COD_DEPE_GROUP_MAP = {
    1: "PUBLICO_MUNICIPAL", 2: "PUBLICO_MUNICIPAL", 6: "PUBLICO_MUNICIPAL",
    3: "PARTICULAR_SUBVENCIONADO", 5: "PARTICULAR_SUBVENCIONADO",
    4: "PARTICULAR_PAGADO",
}

# ----------------------------------------------------------------------------
# Dominios válidos (data quality)
# ----------------------------------------------------------------------------
DOMAINS = {
    "QUINTIL_SE4": {1, 2, 3, 4, 5},
    "GENERO": {1, 2},
    "OBTUVO_BENEFICIO": {0, 1},
}
