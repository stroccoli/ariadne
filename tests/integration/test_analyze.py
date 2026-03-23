from __future__ import annotations

from tests.integration.conftest import _make_incident_state


VALID_PAYLOAD = {"logs": "ERROR: connection refused on db-primary", "mode": "detailed"}


class TestPostAnalyzeHappyPath:
    """Happy-path tests for POST /analyze."""

    def test_post_analyze_valid_detailed_mode(self, client, mock_run_graph):
        """Verify a valid detailed-mode request returns 200 with full AnalyzeResponse."""
        response = client.post("/analyze", json=VALID_PAYLOAD)

        assert response.status_code == 200
        body = response.json()
        assert body["incident_type"] == "database_issue"
        assert body["root_cause"] == "Primary database connection pool exhausted"
        assert 0.0 <= body["confidence"] <= 1.0
        assert isinstance(body["recommended_actions"], list)
        assert len(body["recommended_actions"]) > 0
        assert "metadata" in body

    def test_post_analyze_valid_compact_mode(self, client, mock_run_graph):
        """Verify compact mode is forwarded to run_graph."""
        mock_run_graph.return_value = _make_incident_state(mode="compact")

        response = client.post("/analyze", json={"logs": "timeout observed", "mode": "compact"})

        assert response.status_code == 200
        mock_run_graph.assert_called_once_with(logs="timeout observed", mode="compact")

    def test_post_analyze_default_mode(self, client, mock_run_graph):
        """Verify mode defaults to 'detailed' when omitted."""
        response = client.post("/analyze", json={"logs": "some error logs"})

        assert response.status_code == 200
        mock_run_graph.assert_called_once_with(logs="some error logs", mode="detailed")

    def test_post_analyze_response_metadata(self, client, mock_run_graph):
        """Verify metadata fields are correctly mapped from pipeline state."""
        state = _make_incident_state(
            retrieval_attempts=2,
            total_llm_calls=5,
            total_prompt_tokens=2000,
            total_completion_tokens=600,
            node_timings={"classify": 0.1, "analyze": 0.5, "total": 0.8},
        )
        mock_run_graph.return_value = state

        response = client.post("/analyze", json=VALID_PAYLOAD)

        assert response.status_code == 200
        meta = response.json()["metadata"]
        assert meta["retrieval_attempts"] == 2
        assert meta["llm_calls"] == 5
        assert meta["node_timings"]["classify"] == 0.1
        assert meta["usage"]["prompt_tokens"] == 2000
        assert meta["usage"]["completion_tokens"] == 600
        assert meta["usage"]["total_tokens"] == 2600

    def test_post_analyze_extra_fields_ignored(self, client, mock_run_graph):
        """Verify extra fields in the request body are silently ignored."""
        payload = {**VALID_PAYLOAD, "unexpected_field": "value"}

        response = client.post("/analyze", json=payload)

        assert response.status_code == 200


class TestPostAnalyzeValidationErrors:
    """Tests for 422 validation errors on POST /analyze."""

    def test_post_analyze_missing_logs_returns_422(self, client, mock_run_graph):
        """Verify empty body without 'logs' returns 422."""
        response = client.post("/analyze", json={})

        assert response.status_code == 422
        mock_run_graph.assert_not_called()

    def test_post_analyze_empty_logs_returns_422(self, client, mock_run_graph):
        """Verify empty string for 'logs' fails min_length=1 validation."""
        response = client.post("/analyze", json={"logs": ""})

        assert response.status_code == 422
        mock_run_graph.assert_not_called()

    def test_post_analyze_invalid_mode_returns_422(self, client, mock_run_graph):
        """Verify an invalid mode value is rejected by the Literal constraint."""
        response = client.post("/analyze", json={"logs": "some logs", "mode": "invalid"})

        assert response.status_code == 422
        mock_run_graph.assert_not_called()

    def test_post_analyze_wrong_content_type_returns_422(self, client, mock_run_graph):
        """Verify non-JSON content type is rejected."""
        response = client.post("/analyze", content="not json", headers={"Content-Type": "text/plain"})

        assert response.status_code == 422
        mock_run_graph.assert_not_called()

    def test_post_analyze_null_logs_returns_422(self, client, mock_run_graph):
        """Verify null value for 'logs' returns 422."""
        response = client.post("/analyze", json={"logs": None})

        assert response.status_code == 422
        mock_run_graph.assert_not_called()


class TestPostAnalyzePipelineErrors:
    """Tests for 500 errors when the pipeline fails."""

    def test_post_analyze_pipeline_exception_returns_500(self, client, mock_run_graph):
        """Verify a pipeline RuntimeError produces a 500 with descriptive detail."""
        mock_run_graph.side_effect = RuntimeError("LLM unavailable")

        response = client.post("/analyze", json=VALID_PAYLOAD)

        assert response.status_code == 500
        assert response.json()["detail"] == "Analysis pipeline failed"

    def test_post_analyze_no_output_returns_500(self, client, mock_run_graph):
        """Verify a pipeline that returns no final_output produces a 500."""
        mock_run_graph.return_value = _make_incident_state(final_output=None)

        response = client.post("/analyze", json=VALID_PAYLOAD)

        assert response.status_code == 500
        assert response.json()["detail"] == "Pipeline produced no output"


class TestAnalyzeMethodNotAllowed:
    """Tests for unsupported HTTP methods on /analyze."""

    def test_get_analyze_returns_405(self, client):
        """Verify GET /analyze is not allowed."""
        response = client.get("/analyze")

        assert response.status_code == 405

    def test_put_analyze_returns_405(self, client):
        """Verify PUT /analyze is not allowed."""
        response = client.put("/analyze", json=VALID_PAYLOAD)

        assert response.status_code == 405

    def test_delete_analyze_returns_405(self, client):
        """Verify DELETE /analyze is not allowed."""
        response = client.delete("/analyze")

        assert response.status_code == 405
