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


def test_api_predict_with_fallback(client):
    """
    Verifies that low-confidence / ambiguous tickets activate the fallback in the API.
    """
    payload = {
        "description": "application is not working properly",
        "confidence_threshold": 40.0
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_low_confidence"] is True
    assert data["routing_category"] == "Needs Clarification / Manual Review"
    assert data["triage_status"] == "Manual Triage Required"
    assert "Tier-1 Support Desk" in data["recommended_action"]


def test_confidence_threshold_fallback_logic():
    """
    Directly tests predict_ticket fallback behavior on vague descriptions.
    """
    res = predict_ticket("application is not working properly", confidence_threshold=40.0)
    assert res["is_low_confidence"] is True
    assert res["routing_category"] == "Needs Clarification / Manual Review"
    assert res["triage_status"] == "Manual Triage Required"
    assert "Tier-1" in res["recommended_action"]
    assert "not contain sufficient technical details" in res["suggested_response"]


def test_confidence_threshold_custom_bounds():
    """
    Tests dynamic threshold adjustment (permissive vs strict).
    """
    text = "application is not working properly"
    # Permissive threshold: 20% should NOT trigger fallback since confidence is ~33.7%
    res_permissive = predict_ticket(text, confidence_threshold=20.0)
    assert res_permissive["is_low_confidence"] is False
    assert res_permissive["triage_status"] == "Automated Dispatch"

    # Strict threshold: 60% MUST trigger fallback
    res_strict = predict_ticket(text, confidence_threshold=60.0)
    assert res_strict["is_low_confidence"] is True
    assert res_strict["triage_status"] == "Manual Triage Required"


def test_api_root_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Customer Support Ticket Classifier" in response.text
