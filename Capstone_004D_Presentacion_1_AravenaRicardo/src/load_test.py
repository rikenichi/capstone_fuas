import time
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


BASE_URL = "https://capstone-fuas.onrender.com"
TIMEOUT = 120


PREDICT_PAYLOAD = {
    "NEM": 650,
    "QUINTIL_SE4": 2,
    "COD_DEPE": 3,
    "EDAD": 19,
    "GENERO": 2,
    "NACIONALIDAD": 0
}


def percentile(values, p):
    values = sorted(values)

    if not values:
        return None

    index = int(round((len(values) - 1) * p))
    return values[index]


def make_request(method, endpoint, payload=None):
    start = time.perf_counter()

    try:
        if method == "GET":
            response = requests.get(
                f"{BASE_URL}{endpoint}",
                timeout=TIMEOUT
            )
        else:
            response = requests.post(
                f"{BASE_URL}{endpoint}",
                json=payload,
                timeout=TIMEOUT
            )

        elapsed = time.perf_counter() - start

        return {
            "status": response.status_code,
            "elapsed": elapsed,
            "error": None,
        }

    except requests.RequestException as exc:
        elapsed = time.perf_counter() - start

        return {
            "status": None,
            "elapsed": elapsed,
            "error": str(exc),
        }


def run_test(
    name,
    method,
    endpoint,
    total_requests,
    concurrency,
    payload=None,
    allow_rate_limit=False,
):
    print(f"\n=== {name} ===")
    print("Solicitudes:", total_requests)
    print("Concurrencia:", concurrency)

    start_total = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                make_request,
                method,
                endpoint,
                payload
            )
            for _ in range(total_requests)
        ]

        results = [
            future.result()
            for future in as_completed(futures)
        ]

    total_elapsed = time.perf_counter() - start_total

    status_counts = Counter(
        result["status"]
        for result in results
        if result["status"] is not None
    )

    successful = [
        result
        for result in results
        if result["status"] == 200
    ]

    rate_limited = [
        result
        for result in results
        if result["status"] == 429
    ]

    technical_failures = [
        result
        for result in results
        if (
            result["status"] not in (200, 429)
            or result["error"] is not None
        )
    ]

    success_times = [
        result["elapsed"]
        for result in successful
    ]

    print("\nDistribución HTTP:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    no_status = sum(
        1 for result in results
        if result["status"] is None
    )

    if no_status:
        print(f"  sin respuesta HTTP: {no_status}")

    print("\nResultados:")
    print("Éxitos 200:", len(successful))
    print("Rate limit 429:", len(rate_limited))
    print("Fallos técnicos:", len(technical_failures))

    print(
        "Tasa de éxito 200:",
        f"{len(successful) / len(results) * 100:.2f}%"
    )

    if allow_rate_limit:
        accepted_or_protected = (
            len(successful) + len(rate_limited)
        )

        print(
            "Tasa controlada (200 + 429):",
            f"{accepted_or_protected / len(results) * 100:.2f}%"
        )

    if success_times:
        print("\nLatencia de respuestas 200:")
        print(
            "Media:",
            f"{statistics.mean(success_times):.3f} s"
        )
        print(
            "Mediana:",
            f"{statistics.median(success_times):.3f} s"
        )
        print(
            "P95:",
            f"{percentile(success_times, 0.95):.3f} s"
        )
        print(
            "Máxima:",
            f"{max(success_times):.3f} s"
        )

    print(
        "\nDuración total:",
        f"{total_elapsed:.3f} s"
    )

    print(
        "Throughput total aproximado:",
        f"{len(results) / total_elapsed:.2f} req/s"
    )

    if technical_failures:
        print("\nFallos técnicos detectados:")

        for result in technical_failures[:10]:
            print(
                "status=",
                result["status"],
                "error=",
                result["error"]
            )

    if technical_failures:
        return False

    if rate_limited and not allow_rate_limit:
        return False

    return True


def main():
    print("Load test controlado")
    print("Servidor:", BASE_URL)

    health_ok = run_test(
        name="GET /health",
        method="GET",
        endpoint="/health",
        total_requests=40,
        concurrency=5,
        allow_rate_limit=False,
    )

    predict_ok = run_test(
        name="POST /predict",
        method="POST",
        endpoint="/predict",
        total_requests=20,
        concurrency=2,
        payload=PREDICT_PAYLOAD,
        allow_rate_limit=True,
    )

    print("\n=== RESULTADO GLOBAL ===")

    if health_ok and predict_ok:
        print(
            "PASS - Sin fallos técnicos. "
            "Los 429, si aparecen, corresponden al rate limiter."
        )
        return 0

    print(
        "WARN - Se detectaron fallos técnicos "
        "o comportamiento inesperado."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
