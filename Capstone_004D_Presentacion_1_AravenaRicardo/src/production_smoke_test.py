import sys
import time
import requests


BASE_URL = "https://capstone-fuas.onrender.com"

ENDPOINTS = [
    "/health",
    "/model-info",
    "/model-version",
    "/model-monitoring",
    "/model-policy",
]

MAX_RETRIES = 3
TIMEOUT_SECONDS = 120
WAIT_SECONDS = 10


def check_endpoint(endpoint: str) -> bool:
    url = BASE_URL + endpoint

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=TIMEOUT_SECONDS)

            if response.status_code == 200:
                print(f"OK  {endpoint} -> 200")
                return True

            print(
                f"WARN {endpoint} -> {response.status_code} "
                f"(intento {attempt}/{MAX_RETRIES})"
            )

        except requests.RequestException as exc:
            print(
                f"WARN {endpoint} -> {exc} "
                f"(intento {attempt}/{MAX_RETRIES})"
            )

        if attempt < MAX_RETRIES:
            time.sleep(WAIT_SECONDS)

    print(f"FAIL {endpoint}")
    return False


def main() -> int:
    print(f"Smoke test de producción: {BASE_URL}\n")

    results = {
        endpoint: check_endpoint(endpoint)
        for endpoint in ENDPOINTS
    }

    failed = [endpoint for endpoint, ok in results.items() if not ok]

    print("\nResumen:")
    for endpoint, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'} {endpoint}")

    if failed:
        print("\nSmoke test fallido.")
        print("Endpoints con problemas:", ", ".join(failed))
        return 1

    print("\nSmoke test aprobado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
