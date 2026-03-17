import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from typing import Dict, Any, List

class SIRPredictor:
    """
    Mechanistic Inference Engine (Repurposed from ORDINAR).
    Uses SIR differential equations calibrated against real-time API data.
    """
    
    def __init__(self, population: int = 8000000000):
        self.N = population # Global population default

    def sir_model(self, y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / self.N
        dIdt = beta * S * I / self.N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    def fit_and_predict(self, historical_cases: List[int], days_to_predict: int = 7) -> Dict[str, Any]:
        """
        Calibrates SIR parameters based on historical case counts.
        """
        # Time points for historical data
        t = np.arange(len(historical_cases))
        I0 = historical_cases[0]
        S0 = self.N - I0
        R0 = 0
        y0 = [S0, I0, R0]

        def objective(params):
            beta, gamma = params
            ret = odeint(self.sir_model, y0, t, args=(beta, gamma))
            I_pred = ret[:, 1]
            return np.sum((I_pred - historical_cases)**2)

        # Initial guess for beta (transmission) and gamma (recovery)
        initial_guess = [0.2, 0.1]
        bounds = [(0, 1), (0, 0.5)]
        
        result = minimize(objective, initial_guess, bounds=bounds)
        best_beta, best_gamma = result.x

        # Predict future
        t_future = np.arange(len(historical_cases) + days_to_predict)
        prediction_result = odeint(self.sir_model, y0, t_future, args=(best_beta, best_gamma))
        
        future_cases = prediction_result[len(historical_cases):, 1].tolist()
        
        return {
            "parameters": {
                "beta_transmission": float(best_beta),
                "gamma_recovery": float(best_gamma),
                "r_naught": float(best_beta / best_gamma) if best_gamma != 0 else 0
            },
            "historical_fit": prediction_result[:len(historical_cases), 1].tolist(),
            "prediction_next_7_days": future_cases,
            "engine_status": "ORDINAR Mechanistic Physics-Informed Synthesis Complete"
        }

if __name__ == "__main__":
    # Mock historical data (7 days of growing cases)
    mock_data = [1000, 1500, 2200, 3100, 4500, 6000, 8200]
    predictor = SIRPredictor()
    result = predictor.fit_and_predict(mock_data)
    print(result)
