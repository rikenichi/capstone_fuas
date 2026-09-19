from rate_limit import InMemoryRateLimiter


def test_rate_limiter_blocks_after_limit():
    limiter = InMemoryRateLimiter(
        limits={"/predict": 2},
        window_seconds=60,
    )

    client = "test-client"

    allowed_1, _ = limiter.is_allowed(
        client,
        "/predict",
        now=100.0,
    )

    allowed_2, _ = limiter.is_allowed(
        client,
        "/predict",
        now=101.0,
    )

    allowed_3, retry_after = limiter.is_allowed(
        client,
        "/predict",
        now=102.0,
    )

    assert allowed_1 is True
    assert allowed_2 is True

    assert allowed_3 is False
    assert retry_after > 0


def test_rate_limiter_resets_after_window():
    limiter = InMemoryRateLimiter(
        limits={"/predict": 1},
        window_seconds=60,
    )

    client = "test-client"

    allowed_1, _ = limiter.is_allowed(
        client,
        "/predict",
        now=100.0,
    )

    allowed_2, _ = limiter.is_allowed(
        client,
        "/predict",
        now=161.0,
    )

    assert allowed_1 is True
    assert allowed_2 is True


def test_unprotected_endpoint_is_not_limited():
    limiter = InMemoryRateLimiter(
        limits={"/predict": 1},
        window_seconds=60,
    )

    for _ in range(100):
        allowed, _ = limiter.is_allowed(
            "test-client",
            "/health",
            now=100.0,
        )

        assert allowed is True
