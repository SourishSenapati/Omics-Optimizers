import os
import sys
from typing import Dict, Any

import requests
import torch
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

# Resolve backend path for terminal execution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ingestion import DiseaseHarmonizer
from core.pinn_engine import PINNEngine
from therapeutics_agent import TherapeuticsAgent

app = FastAPI(title="Epidemiological Twin - Mechanistic Inference Core")

# Enable CORS for the static frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module initialization
harmonizer = DiseaseHarmonizer()
pinn_engine = PINNEngine(population=8000000000)
therapeutics_agent = TherapeuticsAgent()

# Static file serving
static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "static")
app.mount("/assets", StaticFiles(directory=static_path), name="static")

@app.get("/")
async def serve_index():
    """Serves the primary web dashboard entry point."""
    return FileResponse(os.path.join(static_path, "index.html"))

@app.get("/heartbeat")
async def heartbeat():
    """Returns engine and hardware status."""
    return {
        "status": "online",
        "cuda": torch.cuda.is_available(),
        "device": str(next(pinn_engine.model.parameters()).device) if hasattr(pinn_engine, "model") else "cpu"
    }

@app.get("/forensic_feed")
async def get_forensic_feed():
    """Provides news-based intelligence feed."""
    return await run_in_threadpool(harmonizer.fetch_promed_alerts)

@app.post("/train")
async def train_pinn(config: Dict[str, Any]):
    """Calibrates and forecasts the epidemic manifold."""
    epochs = config.get("epochs", 10000)
    intervention = config.get("intervention_factor", 0.0)

    try:
        # Standard COVID-19 data source for calibration
        hist_url = "https://disease.sh/v3/covid-19/historical/all?lastdays=60"
        hist_response = await run_in_threadpool(lambda: requests.get(hist_url, timeout=15).json())
        
        raw_cases = list(hist_response['cases'].values())
        raw_recovered = list(hist_response['recovered'].values())
        raw_deaths = list(hist_response['deaths'].values())

        # Calculate active infections for the training manifold
        active = [c - r - d for c, r, d in zip(raw_cases, raw_recovered, raw_deaths)]
        pop = 8000000000

        # Data prep for PyTorch
        s_vals = torch.tensor([(pop - c) / pop for c in raw_cases], dtype=torch.float32)
        i_vals = torch.tensor([i / pop for i in active], dtype=torch.float32)
        r_vals = torch.tensor([r / pop for r in raw_recovered], dtype=torch.float32)

        # Calibration (Wrapped in threadpool to prevent sticking)
        await run_in_threadpool(pinn_engine.train, s_vals, i_vals, r_vals, epochs=epochs)

        # Projection
        results = pinn_engine.get_forecast(days_past=len(raw_cases), intervention=intervention)
        
        # Slicing the already-scaled 'primary' curve into fitting and forecasting parts
        results["historical_fit"] = results["primary"][:len(raw_cases)]
        results["prediction_next_7_days"] = results["primary"][len(raw_cases):len(raw_cases)+7]

        return {"status": "success", "results": results}
        
    except Exception as e:
        print(f"[ENGINE ERROR] Inference cycle failed: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/surveillance")
async def surveillance(disease: str = "COVID-19"):
    """
    DEPRECATED: Support for legacy surveillance calls.
    New dashboard uses the /train and /forensic_feed endpoints.
    """
    data = await run_in_threadpool(harmonizer.harmonize, disease)
    return data

@app.get("/api/therapeutics")
async def get_therapeutics(drug: str = "Dexamethasone"):
    """Provides therapeutic information for a given drug or pathogen."""
    data = await run_in_threadpool(therapeutics_agent.query_drug_mechanism, drug)
    return data


@app.on_event("startup")
async def startup_event():
    """Ensure hardware acceleration is verified on startup."""
    if torch.cuda.is_available():
        print("[FASTAPI] Hardware acceleration verified (CUDA).")
    else:
        print("[FASTAPI] CPU fallback mode active.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
