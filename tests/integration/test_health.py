from __future__ import annotations


class TestGetHealth:
    """Tests for the GET /health endpoint."""

    def test_get_health_returns_ok(self, client):
        """Verify /health returns 200 with status ok."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_get_health_returns_json_content_type(self, client):
        """Verify /health response has application/json content type."""
        response = client.get("/health")

        assert "application/json" in response.headers["content-type"]


class TestGetReady:
    """Tests for the GET /ready endpoint."""

    def test_get_ready_returns_ready(self, client):
        """Verify /ready returns 200 with status ready."""
        response = client.get("/ready")

        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

    def test_get_ready_returns_json_content_type(self, client):
        """Verify /ready response has application/json content type."""
        response = client.get("/ready")

        assert "application/json" in response.headers["content-type"]
