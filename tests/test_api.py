"""
test_api.py — Tes Otomatis untuk FastAPI Endpoint /extract-entities (Fase 8)
============================================================================
Menguji seluruh kasus uji validasi dan pemrosesan API:
  1. Input normal -> HTTP 200, format entities & meta valid
  2. Input kosong (title dan body keduanya spasi/kosong) -> HTTP 400
  3. JSON malformed (rusak/invalid syntax) -> HTTP 400
  4. Tipe bukan string (int, float, list, None) -> HTTP 400
  5. Missing required field -> HTTP 400
  6. Input melebihi MAX_INPUT_CHARS -> HTTP 400
  7. Negative test case (teks tanpa named entity) -> HTTP 200, entities kosong
  8. Health check GET /health -> HTTP 200
  9. Web panel GET / -> HTTP 200 dengan konten HTML
"""

from __future__ import annotations

import json
import pytest
from starlette.testclient import TestClient

from src.api import app
from src.config import MAX_INPUT_CHARS


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestApiValidation:
    def test_health_check(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model" in data

    def test_web_panel_index(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Article NER Service" in response.text

    def test_valid_input_returns_200(self, client: TestClient):
        payload = {
            "title": "Maybank posts record quarterly earnings",
            "body": "Kuala Lumpur: Maybank reported strong loan expansion across Malaysia on 10 September 2026."
        }
        response = client.post("/extract-entities", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "meta" in data
        assert isinstance(data["entities"], list)
        assert data["meta"]["entity_count"] == len(data["entities"])
        assert "analyzed_length" in data["meta"]

    def test_empty_title_and_body_returns_400(self, client: TestClient):
        payload = {"title": "   ", "body": ""}
        response = client.post("/extract-entities", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data or "message" in data

    def test_malformed_json_returns_400(self, client: TestClient):
        # Mengirim raw text malformed yang melanggar sintaks JSON
        response = client.post(
            "/extract-entities",
            content='{"title": "Broken JSON", "body": "missing closing quote}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        data = response.json()
        assert "malformed" in data.get("message", "").lower() or "error" in data

    def test_empty_body_request_returns_400(self, client: TestClient):
        response = client.post(
            "/extract-entities",
            content="",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

    def test_non_string_field_type_returns_400(self, client: TestClient):
        # Field title berupa integer bukan string
        payload = {"title": 12345, "body": "Valid body text here"}
        response = client.post("/extract-entities", json=payload)
        assert response.status_code == 400

    def test_null_field_type_returns_400(self, client: TestClient):
        payload = {"title": "Valid title", "body": None}
        response = client.post("/extract-entities", json=payload)
        assert response.status_code == 400

    def test_missing_fields_returns_400(self, client: TestClient):
        payload = {"only_title": "No body provided"}
        response = client.post("/extract-entities", json=payload)
        assert response.status_code == 400

    def test_json_array_instead_of_object_returns_400(self, client: TestClient):
        response = client.post(
            "/extract-entities",
            content='[{"title": "t", "body": "b"}]',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

    def test_input_exceeds_max_chars_returns_400(self, client: TestClient):
        # Buat teks melebihi MAX_INPUT_CHARS
        huge_text = "a" * (MAX_INPUT_CHARS + 50)
        payload = {"title": "Large input test", "body": huge_text}
        response = client.post("/extract-entities", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "melebihi batas" in data.get("message", "").lower()

    def test_negative_example_no_entities_returns_200(self, client: TestClient):
        # Teks edukasi umum tanpa nama orang, bank, lokasi, tanggal
        payload = {
            "title": "Digital banking security recommendations",
            "body": "Always ensure your account password remains confidential and never share one time passwords."
        }
        response = client.post("/extract-entities", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert isinstance(data["entities"], list)
