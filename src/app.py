"""
FastAPI Web Application for Customer Support Ticket Classification.
Assignment Task 6 - Option B.

Runs 100% locally with zero external API calls.
Serves static frontend and RESTful inference endpoints.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from src.predict import predict_ticket, load_model, BENCHMARK_TEST_CASES


app = FastAPI(
    title="Support Ticket Intelligence API",
    description="Offline Machine Learning Ticket Categorization & Automated Response System",
    version="1.0.0"
)

# Request & Response Schemas
class TicketRequest(BaseModel):
    description: str = Field(..., min_length=2, description="Customer ticket description")
    customer_name: Optional[str] = Field(None, description="Optional customer name for personalization")
    priority: Optional[str] = Field("Medium", description="Ticket priority (Low, Medium, High, Critical)")


class TicketResponse(BaseModel):
    ticket_description: str
    cleaned_description: str
    predicted_category: str
    confidence: float
    all_probabilities: dict
    suggested_response: str
    recommended_action: str
    estimated_sla: str


# Mount static assets
STATIC_DIR = PROJECT_ROOT / "static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serves the primary web interface."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend interface not found.")
    return FileResponse(index_file)


@app.get("/api/health")
async def health_check():
    """Returns system status, active model information, and API-free guarantee."""
    try:
        model_payload = load_model()
        return {
            "status": "healthy",
            "model_name": model_payload.get("model_name", "Logistic Regression"),
            "categories_supported": len(model_payload.get("classes", [])),
            "offline_mode": True,
            "external_api_calls": 0
        }
    except Exception as exc:
        return {"status": "degraded", "error": str(exc)}


@app.get("/api/samples")
async def get_sample_tickets():
    """Returns curated benchmark tickets for rapid UI testing."""
    return BENCHMARK_TEST_CASES


@app.post("/api/predict", response_model=TicketResponse)
async def api_predict(ticket: TicketRequest):
    """
    Classifies a ticket description into one of 8 categories with confidence score
    and offline automated response.
    """
    if not ticket.description.strip():
        raise HTTPException(status_code=400, detail="Ticket description cannot be empty.")

    try:
        result = predict_ticket(
            ticket_description=ticket.description,
            customer_name=ticket.customer_name,
            priority=ticket.priority or "Medium"
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Starts local Uvicorn ASGI server."""
    print("=" * 65)
    print("  CUSTOMER SUPPORT TICKET INTELLIGENCE SYSTEM - WEB INTERFACE")
    print(f"  Access Web UI at: http://{host}:{port}")
    print("  100% Local Execution | Zero External APIs")
    print("=" * 65)
    uvicorn.run("src.app:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start Support Ticket Web Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)
