import asyncio
import json
import os
import sys
import requests
import torch

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ingestion import DiseaseHarmonizer
from core.pinn_engine import PINNEngine

async def generate_payload():
    print("[PRE-BAKE] Initializing Engines...")
    harmonizer = DiseaseHarmonizer()
    pinn_engine = PINNEngine(population=8000000000)
    
    disease = "COVID-19"
    epochs = 10000
    intervention = 0.1
    
    print(f"[PRE-BAKE] Fetching Global Stats and Alerts for {disease}...")
    stats = harmonizer.fetch_global_stats()
    alerts = harmonizer.fetch_promed_alerts()
    
    print(f"[PRE-BAKE] Starting PINN Training ({epochs} iterations)...")
    # Fetch historical data for training
    hist_url = "https://disease.sh/v3/covid-19/historical/all?lastdays=60"
    hist_response = requests.get(hist_url, timeout=15).json()
    
    raw_cases = list(hist_response['cases'].values())
    raw_recovered = list(hist_response['recovered'].values())
    raw_deaths = list(hist_response['deaths'].values())
    active = [c - r - d for c, r, d in zip(raw_cases, raw_recovered, raw_deaths)]
    pop = 8000000000

    s_vals = torch.tensor([(pop - c) / pop for c in raw_cases], dtype=torch.float32)
    i_vals = torch.tensor([i / pop for i in active], dtype=torch.float32)
    r_vals = torch.tensor([r / pop for r in raw_recovered], dtype=torch.float32)

    # Train
    pinn_engine.train(s_vals, i_vals, r_vals, epochs=epochs, print_freq=2000)
    
    print("[PRE-BAKE] Generating Forecast...")
    modeling = pinn_engine.get_forecast(days_past=len(raw_cases), intervention=intervention)
    
    # Structure the payload as expected by the frontend
    payload = {
        "metadata": {
            "source": "Global Surveillance (Pre-Baked)",
            "target": disease,
            "timestamp": "2026-03-18T15:00:00Z"
        },
        "stats": stats,
        "alerts": alerts,
        "modeling": modeling,
        "status": "success"
    }
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "omics_intelligence_payload.json")
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=4)
    
    print(f"[PRE-BAKE] Payload secured at: {output_path}")

if __name__ == "__main__":
    asyncio.run(generate_payload())
