# Verification script for backend components
# Measures performance and accuracy of the calibration core

import time
import torch
from core.pinn_engine import PINNEngine

def verify_calibration():
    # Performance assessment of the PINN optimizer
    print("--- High-Fidelity Calibration Test ---")
    if torch.cuda.is_available():
        print(f"Hardware: {torch.cuda.get_device_name(0)}")
    else:
        print("Hardware: CPU")
    
    # Engine initialization
    engine = PINNEngine(population=8000000000)
    
    # Dummy data for validation (60-day window)
    s_dummy = torch.linspace(0.99, 0.90, 60)
    i_dummy = torch.linspace(0.01, 0.08, 60)
    r_dummy = torch.linspace(0.0, 0.02, 60)
    
    t_start = time.time()
    # 10,000 cycle training protocol
    engine.train(s_dummy, i_dummy, r_dummy, epochs=10000)
    t_end = time.time()
    
    res = engine.get_forecast(days_past=60, days_future=7)
    kinetics = res['metadata']['inferred_kinetics']
    
    print("\n[PERFORMANCE RESULTS]")
    print(f"Time for 10k Cycles: {t_end - t_start:.2f} seconds")
    print(f"Beta Estimation: {kinetics['beta']:.4f}")
    print(f"R0 Estimation: {kinetics['r_naught']:.2f}")
    
    print("\n--- Calibration Test Complete ---")

if __name__ == "__main__":
    verify_calibration()
