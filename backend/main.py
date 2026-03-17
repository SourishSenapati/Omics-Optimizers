# Main API gateway
# Final technical version for March 18th

import requests
import uvicorn
import torch
from fastapi import FastAPI
from core.ingestion import DiseaseHarmonizer
from core.pinn_engine import PINNEngine

app = FastAPI(title="Epidemiological Twin - Mechanistic Inference Core")

# Module initialization
harmonizer = DiseaseHarmonizer()
pinn_engine = PINNEngine(population=8000000000)

@app.get("/")
async def status():
    # Health check endpoint
    return {
        "status": "online",
        "timestamp": "2026-03-17",
        "version": "1.0.0"
    }

@app.get("/api/surveillance")
async def surveillance(disease: str = "COVID-19", epochs: int = 10000):
    # API for data ingestion and high-fidelity modeling
    data = harmonizer.harmonize(disease)
    
    try:
        # Fetch historical cases for calibration
        hist_url = "https://disease.sh/v3/covid-19/historical/all?lastdays=60"
        hist_response = requests.get(hist_url, timeout=15).json()
        
        raw_cases = list(hist_response['cases'].values())
        raw_recovered = list(hist_response['recovered'].values())
        raw_deaths = list(hist_response['deaths'].values())
        
        # Calculate active infections for the training manifold
        active = [c - r - d for c, r, d in zip(raw_cases, raw_recovered, raw_deaths)]
        
        # Modeling parameters
        pop = 8000000000
        s_vals = torch.tensor([(pop - c) / pop for c in raw_cases], dtype=torch.float32)
        i_vals = torch.tensor([i / pop for i in active], dtype=torch.float32)
        r_vals = torch.tensor([r / pop for r in raw_recovered], dtype=torch.float32)
        
        # PINN training/calibration
        pinn_engine.train(s_vals, i_vals, r_vals, epochs=epochs)
        
        # Model projection
        forecast_res = pinn_engine.get_forecast(days_past=len(raw_cases))
        forecast_res["historical"] = active
        data["modeling"] = forecast_res
        
    except requests.exceptions.RequestException as e:
        data["modeling"] = {"error": f"Connection error: {str(e)}"}
    except (KeyError, ValueError) as e:
        data["modeling"] = {"error": f"Data error: {str(e)}"}
    except Exception as e:
        data["modeling"] = {"error": f"Runtime error: {str(e)}"}
        
    return data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
