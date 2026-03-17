# Integrated modeling core with logging and V2 capabilities

import time
import os
from typing import Dict, Any
import torch
from torch import nn

# Hardware selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EpiPINN(nn.Module):
    # Network for SIR state approximation
    def __init__(self):
        super(EpiPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 3) 
        )
        self.beta = nn.Parameter(torch.tensor([0.5], device=DEVICE, requires_grad=True))
        self.gamma = nn.Parameter(torch.tensor([0.1], device=DEVICE, requires_grad=True))
        
    def forward(self, t):
        return self.net(t)

class PINNEngine:
    def __init__(self, population: int = 8000000000):
        self.total_pop = population
        self.model = EpiPINN().to(DEVICE)
        self.log_file = os.path.join("docs", "logs", f"training_log_{int(time.time())}.txt")
        self.model_path = os.path.join("docs", "models", "latest_pinn.pt")
        print(f"[ENGINE] Hardware: {DEVICE.type.upper()}")

    def save(self):
        """Persist the neural network state to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"[ENGINE] Checkpoint secured: {self.model_path}")
        
    def train(self, s_data: torch.Tensor, i_data: torch.Tensor, r_data: torch.Tensor, epochs: int = 15000, print_freq: int = 1000):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        t_data = torch.linspace(0, len(s_data) - 1, len(s_data), device=DEVICE).view(-1, 1).requires_grad_(True)
        s_target, i_target, r_target = s_data.view(-1, 1).to(DEVICE), i_data.view(-1, 1).to(DEVICE), r_data.view(-1, 1).to(DEVICE)

        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"--- Training Session Started ({epochs} epochs) ---\n")
            f.write(f"Hardware: {DEVICE.type.upper()}\n")

        try:
            for epoch in range(epochs):
                optimizer.zero_grad()
                preds = self.model(t_data)
                s_p, i_p, r_p = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
                
                ds_dt = torch.autograd.grad(s_p, t_data, torch.ones_like(s_p), create_graph=True)[0]
                di_dt = torch.autograd.grad(i_p, t_data, torch.ones_like(i_p), create_graph=True)[0]
                dr_dt = torch.autograd.grad(r_p, t_data, torch.ones_like(r_p), create_graph=True)[0]
                
                f_s = ds_dt + self.model.beta * s_p * i_p
                f_i = di_dt - self.model.beta * s_p * i_p + self.model.gamma * i_p
                f_r = dr_dt - self.model.gamma * i_p
                
                loss_phys = torch.mean(f_s**2 + f_i**2 + f_r**2)
                loss_data = torch.mean((s_p - s_target)**2 + (i_p - i_target)**2 + (r_p - r_target)**2)
                total_loss = loss_data + 0.1 * loss_phys
                
                total_loss.backward()
                optimizer.step()
                
                if epoch % 100 == 0:
                    log_entry = f"Epoch {epoch} | Loss: {total_loss.item():.8f} | Beta: {self.model.beta.item():.4f} | Gamma: {self.model.gamma.item():.4f}\n"
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(log_entry)
                
                if epoch % print_freq == 0:
                    print(f"Epoch {epoch} | Loss: {total_loss.item():.8f} | Beta: {self.model.beta.item():.4f} | Gamma: {self.model.gamma.item():.4f}")
            
            self.save()
        except KeyboardInterrupt:
            print("\n[ENGINE] Interrupt detected. Finalizing checkpoint...")
            self.save()
                
        return self.model

    def get_forecast(self, days_past: int, days_future: int = 14, intervention: float = 0.0) -> Dict[str, Any]:
        t_all = torch.linspace(0, days_past + days_future - 1, days_past + days_future, device=DEVICE).view(-1, 1)
        learned_beta = self.model.beta.item()
        eff_beta, gamma = learned_beta * (1.0 - intervention), self.model.gamma.item()
        
        with torch.no_grad():
            preds = self.model(t_all)
            full_curve = (preds[:, 1].cpu() * self.total_pop).tolist()
            coupling_coeff = 0.05 
            coupled_threat = [(val * coupling_coeff) for val in full_curve]

        return {
            "metadata": {
                "hardware": DEVICE.type.upper(),
                "kinetics": {"beta": learned_beta, "beta_eff": eff_beta, "gamma": gamma, "r0": eff_beta/gamma},
                "log": self.log_file
            },
            "primary": full_curve,
            "coupled": coupled_threat,
            "status": "Success"
        }
