"""
New Ticket Prediction Pipeline (Assignment Task 5).

Provides inference capabilities for unseen customer tickets:
1. Loads the trained model pipeline (TF-IDF + Classifier).
2. Computes the predicted category and calibrated confidence score (%).
3. Returns ranked probabilities across all 8 support categories.
4. Generates an offline automated resolution response (Bonus GenAI Task).
5. Documents and validates test cases on completely new, unseen inputs.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np

from src.preprocessing import clean_text
from src.responder import generate_automated_response

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ticket_classifier.joblib"
_CACHED_MODEL_PAYLOAD = None


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """
    Loads and caches the serialized model pipeline.
    """
    global _CACHED_MODEL_PAYLOAD
    if _CACHED_MODEL_PAYLOAD is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                f"Please execute 'python src/train.py' first to train the model."
            )
        _CACHED_MODEL_PAYLOAD = joblib.load(model_path)
    return _CACHED_MODEL_PAYLOAD


def predict_ticket(
    ticket_description: str,
    customer_name: Optional[str] = None,
    priority: str = "Medium",
    model_path: Path = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Predicts the category of a support ticket description and calculates
    confidence probability along with an automated resolution suggestion.
    """
    payload = load_model(model_path)
    pipeline = payload["pipeline"]
    classes = payload["classes"]

    # Preprocess text
    cleaned = clean_text(ticket_description)
    if not cleaned:
        return {
            "ticket_description": ticket_description,
            "cleaned_description": "",
            "predicted_category": "Unknown",
            "confidence": 0.0,
            "all_probabilities": {},
            "suggested_response": "Please enter a valid ticket description.",
            "recommended_action": "Request more information from user.",
            "estimated_sla": "N/A",
        }

    # Predict probabilities
    probabilities = pipeline.predict_proba([cleaned])[0]
    best_idx = int(np.argmax(probabilities))
    best_category = classes[best_idx]
    confidence_score = float(probabilities[best_idx]) * 100.0

    # Sort all probabilities descending
    sorted_indices = np.argsort(probabilities)[::-1]
    all_probs = {
        classes[i]: round(float(probabilities[i]) * 100.0, 2)
        for i in sorted_indices
    }

    # Generate offline customer response (Bonus Task)
    response_meta = generate_automated_response(
        category=best_category,
        ticket_description=ticket_description,
        customer_name=customer_name,
        priority=priority,
        confidence=confidence_score,
    )

    return {
        "ticket_description": ticket_description,
        "cleaned_description": cleaned,
        "predicted_category": best_category,
        "confidence": round(confidence_score, 2),
        "all_probabilities": all_probs,
        "suggested_response": response_meta["response_text"],
        "recommended_action": response_meta["recommended_action"],
        "estimated_sla": response_meta["estimated_sla"],
    }


# Documented Test Cases for Assignment Task 5
BENCHMARK_TEST_CASES = [
    {
        "input": "I forgot my password and cannot sign into my account",
        "expected": "Login Issue",
    },
    {
        "input": "The application shows a fatal 500 internal server error when saving",
        "expected": "Application Error",
    },
    {
        "input": "Need the quarterly financial and sales performance report",
        "expected": "Report",
    },
    {
        "input": "I would like to change my registered mobile phone number and email",
        "expected": "Account Update",
    },
    {
        "input": "The web page is taking too long to load and frequently times out",
        "expected": "Performance",
    },
    {
        "input": "My credit card was charged twice but payment is still showing failed",
        "expected": "Payment Issue",
    },
    {
        "input": "Permission denied when trying to view the administrator dashboard",
        "expected": "Access Issue",
    },
    {
        "input": "Several customer records are missing or corrupted after sync",
        "expected": "Data Issue",
    },
]


def run_benchmark_tests():
    """
    Runs and prints documented verification on 8 unseen test tickets.
    """
    print("\n" + "=" * 78)
    print("TASK 5: PREDICTION EVALUATION ON UNSEEN BENCHMARK TEST TICKETS")
    print("=" * 78)
    print(f"{'#':<3} | {'Test Ticket Description':<40} | {'Expected':<18} | {'Predicted':<18} | {'Conf':<6} | {'Status'}")
    print("-" * 105)

    correct = 0
    results = []
    for idx, test_case in enumerate(BENCHMARK_TEST_CASES, start=1):
        pred_result = predict_ticket(test_case["input"])
        predicted = pred_result["predicted_category"]
        conf = pred_result["confidence"]
        is_correct = predicted == test_case["expected"]
        if is_correct:
            correct += 1
        status_symbol = "MATCH [OK]" if is_correct else "MISMATCH [X]"

        # Truncate text for table display if needed
        disp_text = (test_case["input"][:37] + "...") if len(test_case["input"]) > 40 else test_case["input"]
        print(f"{idx:<3} | {disp_text:<40} | {test_case['expected']:<18} | {predicted:<18} | {conf:>5.1f}% | {status_symbol}")
        results.append({
            "test_input": test_case["input"],
            "expected_category": test_case["expected"],
            "predicted_category": predicted,
            "confidence": conf,
            "match": is_correct
        })

    accuracy = (correct / len(BENCHMARK_TEST_CASES)) * 100.0
    print("-" * 105)
    print(f"Benchmark Results: {correct}/{len(BENCHMARK_TEST_CASES)} Correct ({accuracy:.1f}% Accuracy)\n")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Customer Support Ticket Predictor")
    parser.add_argument("--text", type=str, help="Single ticket description to predict")
    parser.add_argument("--test", action="store_true", help="Run documented 8-case benchmark suite")
    parser.add_argument("--priority", type=str, default="Medium", help="Ticket priority (Low, Medium, High, Critical)")
    args = parser.parse_args()

    if args.test or (len(sys.argv) == 1):
        run_benchmark_tests()

    if args.text:
        res = predict_ticket(args.text, priority=args.priority)
        print("=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"Ticket Description : {res['ticket_description']}")
        print(f"Cleaned Text       : {res['cleaned_description']}")
        print(f"Predicted Category : {res['predicted_category']}")
        print(f"Confidence         : {res['confidence']:.1f}%")
        print("\nAll Category Probabilities:")
        for cat, prob in res["all_probabilities"].items():
            print(f"  • {cat:20s}: {prob:5.1f}%")
        print("\nSuggested Automated Response (Offline GenAI):")
        print(res["suggested_response"])
        print("=" * 60)
