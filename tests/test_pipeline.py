"""
Automated Test Suite for Support Ticket Intelligence System.
Verifies preprocessing, training pipeline, prediction accuracy, and FastAPI endpoints.
"""

import sys
from pathlib import Path
import pytest
from starlette.testclient import TestClient

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_text, load_dataset, validate_and_summarize_dataset
from src.predict import predict_ticket, load_model
from src.responder import generate_automated_response
from src.app import app


# -----------------------------------------------------------------------------
# 1. Preprocessing Tests
# -----------------------------------------------------------------------------
def test_clean_text_basic():
    raw = "  I CANNOT login to the application! Please help... "
    cleaned = clean_text(raw)
    assert cleaned == "i cannot login to the application please help"


def test_clean_text_urls_and_symbols():
    raw = "Check https://help.company.com/error#404 or email support@test.com for <alert>!"
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "support@test.com" not in cleaned
    assert "<alert>" not in cleaned


def test_clean_text_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_dataset_validation():
    df = load_dataset("data/customer_support_ticket_dataset_200.csv")
    summary = validate_and_summarize_dataset(df)
    assert summary["total_records"] == 200
    assert len(summary["category_counts"]) == 8
    assert summary["missing_required_columns"] == []


# -----------------------------------------------------------------------------
# 2. Prediction & Model Pipeline Tests
# -----------------------------------------------------------------------------
def test_model_loading():
    payload = load_model()
    assert "pipeline" in payload
    assert "classes" in payload
    assert len(payload["classes"]) == 8


def test_predict_ticket_valid():
    res = predict_ticket("I forgot my password and cannot sign into the system")
    assert res["predicted_category"] == "Login Issue"
    assert 0.0 <= res["confidence"] <= 100.0
    assert len(res["all_probabilities"]) == 8
    assert "suggested_response" in res
    assert "recommended_action" in res


def test_predict_ticket_all_categories():
    test_cases = [
        ("Application gives error while saving data", "Application Error"),
        ("Please help me generate my monthly sales report", "Report"),
        ("I need to change my registered mobile number", "Account Update"),
        ("The application is very slow today", "Performance"),
        ("Payment failed while completing checkout", "Payment Issue"),
        ("I cannot access the admin dashboard", "Access Issue"),
        ("Customer data is missing from the system", "Data Issue"),
    ]
    for text, expected in test_cases:
        res = predict_ticket(text)
        assert res["predicted_category"] == expected, f"Failed for '{text}': got {res['predicted_category']}"


# -----------------------------------------------------------------------------
# 3. Offline Response Generator Tests
# -----------------------------------------------------------------------------
def test_offline_response_generator():
    resp = generate_automated_response(
        category="Payment Issue",
        ticket_description="Money was deducted twice",
        customer_name="Priya",
        priority="Critical"
    )
    assert "Priya" in resp["response_text"]
    assert "Critical Priority SLA" in resp["estimated_sla"]
    assert resp["category"] == "Payment Issue"
    assert resp["recommended_action"] != ""


# -----------------------------------------------------------------------------
# 4. FastAPI Server & API Endpoints Tests
# -----------------------------------------------------------------------------
@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["offline_mode"] is True
    assert data["external_api_calls"] == 0


def test_api_predict_success(client):
    payload = {
        "description": "I need help exporting the quarterly sales numbers report",
        "customer_name": "Amit",
        "priority": "Low"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_category"] == "Report"
    assert data["confidence"] > 0
    assert "Amit" in data["suggested_response"]


def test_api_predict_empty_validation(client):
    response = client.post("/api/predict", json={"description": ""})
    assert response.status_code == 422 or response.status_code == 400


def test_api_root_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Customer Support Ticket Classifier" in response.text
