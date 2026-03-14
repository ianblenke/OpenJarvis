"""Tests for TUI dashboard (Phase 16.5)."""

from __future__ import annotations

import pytest

from openjarvis.cli.dashboard import DashboardApp


class TestDashboard:
    @pytest.mark.spec("REQ-cli.telemetry")
    def test_available_check(self):
        result = DashboardApp.available()
        assert isinstance(result, bool)

    @pytest.mark.spec("REQ-cli.telemetry")
    def test_create_app(self):
        app = DashboardApp()
        assert app is not None

    @pytest.mark.spec("REQ-cli.telemetry")
    def test_create_app_with_config(self):
        app = DashboardApp(config={"test": True})
        assert app._config == {"test": True}

    @pytest.mark.skipif(
        not DashboardApp.available(),
        reason="textual not installed",
    )
    @pytest.mark.spec("REQ-cli.telemetry")
    def test_run_import(self):
        """Verify run method can at least be called (won't actually launch)."""
        # Just test that DashboardApp is properly importable and constructible
        app = DashboardApp()
        assert hasattr(app, "run")
