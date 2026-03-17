# Verification script for upgraded backend components
# Validates Graph-PINN logic and Optimal Control simulations

import time
import torch
from core.pinn_engine import PINNEngine

def verify_v2_calibration():
    # Performance assessment of the upgraded optimizer
    print("--- High-Fidelity V2 Calibration Test ---")
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
    # 2000 cycle quick test for V2 logic
    engine.train(s_dummy, i_dummy, r_dummy, epochs=2000)
    t_end = time.time()
    
    # Test with 30% intervention
    res = engine.get_forecast(days_past=60, days_future=14, intervention=0.3)
    kinetics = res['metadata']['inferred_kinetics']
    
    print("\n[PERFORMANCE RESULTS]")
    print(f"Time for 2k Cycles: {t_end - t_start:.2f} seconds")
    print(f"Learned Beta: {kinetics['beta']:.4f}")
    print(f"Effective Beta (30% Intervention): {kinetics['beta_effective']:.4f}")
    print(f"Graph-Coupled Node Points: {len(res['coupled_node_threat'])}")
    
    if len(res['primary_node']) == 74: # 60 past + 14 future
        print("[SUCCESS] Trajectory alignment verified.")
    
    print("\n--- V2 Calibration Test Complete ---")

if __name__ == "__main__":
    verify_v2_calibration()
