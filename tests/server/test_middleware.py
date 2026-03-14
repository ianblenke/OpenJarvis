"""Tests for security middleware -- HTTP security headers."""

from __future__ import annotations

import sys

import pytest

from openjarvis.server.middleware import SECURITY_HEADERS, create_security_middleware


class TestSecurityHeaders:
    """Tests for security headers middleware."""

    @pytest.mark.spec("REQ-server.middleware.security")
    def test_headers_dict(self) -> None:
        """Verify SECURITY_HEADERS has all expected keys."""
        expected_keys = {
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "Referrer-Policy",
            "Permissions-Policy",
        }
        assert set(SECURITY_HEADERS.keys()) == expected_keys

    @pytest.mark.spec("REQ-server.middleware.security")
    def test_create_middleware_without_starlette(self, monkeypatch) -> None:
        """When starlette is not available, returns None."""
        import importlib

        import openjarvis.server.middleware as mod

        blocked_keys = [
            "starlette",
            "starlette.middleware",
            "starlette.middleware.base",
            "starlette.requests",
            "starlette.responses",
        ]
        for key in blocked_keys:
            monkeypatch.setitem(sys.modules, key, None)
        try:
            importlib.reload(mod)
            result = mod.create_security_middleware()
            assert result is None
        finally:
            # Reload to restore normal state (monkeypatch reverts sys.modules)
            for key in blocked_keys:
                sys.modules.pop(key, None)
            importlib.reload(mod)

    @pytest.mark.spec("REQ-server.middleware.security")
    def test_create_middleware_with_starlette(self) -> None:
        """When starlette is available, returns a class."""
        middleware_cls = create_security_middleware()
        if middleware_cls is None:
            # starlette not installed -- skip
            import pytest
            pytest.skip("starlette not available")
        assert middleware_cls is not None
        assert callable(middleware_cls)

    @pytest.mark.spec("REQ-server.middleware.security")
    def test_middleware_adds_headers(self) -> None:
        """Middleware adds all security headers to responses."""
        import pytest
        fastapi = pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        app = fastapi.FastAPI()

        middleware_cls = create_security_middleware()
        assert middleware_cls is not None
        app.add_middleware(middleware_cls)

        @app.get("/test")
        @pytest.mark.spec("REQ-server.middleware.security")
        def test_endpoint() -> dict:
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200

        for header_name, header_value in SECURITY_HEADERS.items():
            assert resp.headers.get(header_name) == header_value, (
                f"Missing or wrong header: {header_name}"
            )


class TestCorsMiddleware:
    """Tests for CORS middleware configuration."""

    @pytest.mark.spec("REQ-server.middleware.cors")
    def test_cors_middleware_configured(self) -> None:
        """CORS middleware is added to the app during create_app."""
        pytest.importorskip("fastapi")

        from openjarvis.server.app import create_app
        from tests.fixtures.engines import FakeEngine

        engine = FakeEngine()
        app = create_app(engine, "test-model")

        # Check that CORSMiddleware is in the middleware stack
        # FastAPI stores middleware as Middleware objects
        has_cors = any("CORS" in str(m) for m in app.user_middleware)
        assert has_cors, "CORSMiddleware should be configured"
