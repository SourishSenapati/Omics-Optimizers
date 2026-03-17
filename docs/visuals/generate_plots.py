import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Professional plot styling for high-stakes presentation
plt.style.use('dark_background')
colors = {'S': '#3b82f6', 'I': '#ef4444', 'R': '#10b981'}
font_cfg = {'family': 'sans-serif', 'weight': 'bold', 'size': 12}

def sir_model(y, t, beta, gamma):
    # SIR Ordinary Differential Equations
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def generate_sir_curve():
    # Parameters for a significant outbreak
    N = 1.0
    beta, gamma = 0.35, 0.1
    y0 = [0.99, 0.01, 0.0]
    t = np.linspace(0, 100, 500)
    
    # Solve trajectory
    ret = odeint(sir_model, y0, t, args=(beta, gamma))
    S, I, R = ret.T
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, S, label='Susceptible', color=colors['S'], lw=3)
    ax.plot(t, I, label='Infected', color=colors['I'], lw=3)
    ax.plot(t, R, label='Recovered', color=colors['R'], lw=3)
    
    ax.set_xlabel('Days Since Outbreak Trace', **font_cfg)
    ax.set_ylabel('Population Fraction', **font_cfg)
    ax.set_title('Epidemic Manifold: Mechanistic SIR Projection', **font_cfg, pad=20)
    ax.legend(frameon=False)
    
    # Grid and aesthetic polish
    ax.grid(color='#374151', linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig('docs/visuals/sir_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_pinn_flowchart():
    # Architectural visualization of the PINN-SINDy pipeline
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Define box properties
    box_props = dict(boxstyle='round,pad=0.8', fc='#1f2937', ec='#4b5563', lw=2)
    text_cfg = {'color': '#f3f4f6', 'weight': 'bold', 'ha': 'center', 'size': 10}
    
    # Data Layer
    ax.text(0.5, 0.9, 'RAW TELEMETRY\n(Disease.sh API / ProMED Feed)', bbox=box_props, **text_cfg)
    ax.annotate('', xy=(0.5, 0.8), xytext=(0.5, 0.88), arrowprops=dict(arrowstyle='->', lw=2, color='#4b5563'))
    
    # Harmonization Layer
    ax.text(0.5, 0.75, 'DATA HARMONIZATION ENGINE\n(Normalization & Noise Reduction)', bbox=box_props, **text_cfg)
    ax.annotate('', xy=(0.5, 0.65), xytext=(0.5, 0.73), arrowprops=dict(arrowstyle='->', lw=2, color='#4b5563'))
    
    # Neural Manifold
    ax.text(0.5, 0.55, 'PHYSICS-INFORMED NEURAL NETWORK\n(3-Layer State Approximation Core)', bbox=box_props, **text_cfg)
    ax.annotate('', xy=(0.5, 0.45), xytext=(0.5, 0.53), arrowprops=dict(arrowstyle='->', lw=2, color='#4b5563'))
    
    # Physics Loss
    ax.text(0.5, 0.35, 'MECHANISTIC CONSTRAINT EVALUATION\n(SIR ODE Residual Calculations)', bbox=box_props, **text_cfg)
    ax.annotate('', xy=(0.5, 0.25), xytext=(0.5, 0.33), arrowprops=dict(arrowstyle='->', lw=2, color='#4b5563'))
    
    # Output Layer
    ax.text(0.5, 0.15, 'PREDICTIVE DIGITAL TWIN OUTPUT\n(Inferred Parameters & Trajectory Forensics)', bbox=box_props, **text_cfg)
    
    ax.set_title('Omics Optimizers Architecture: PINN Inference Flow', **font_cfg, pad=10)
    plt.tight_layout()
    plt.savefig('docs/visuals/architecture_flow.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("[SYSTEM] Synthesizing high-resolution visualizations...")
    generate_sir_curve()
    generate_pinn_flowchart()
    print("[SUCCESS] Graphics rendered to docs/visuals/ directory.")
