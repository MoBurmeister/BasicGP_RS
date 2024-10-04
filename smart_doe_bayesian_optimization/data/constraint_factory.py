import torch
import math
from typing import List, Tuple, Callable


class WeldingConstraints:
    '''
    @matthewcarbone I think you misunderstood how the nonlinear constraint function is 
    supposed to be defined. Note that the constraint function is supposed to return 
    a numerical value, where positive values indicate feasibility and 
    negative values indicate infeasibility 
    (this is consistent with the scipy convention). 
    '''
    
    def __init__(self, P: float, L: float, t_max: float, s_max: float):
        """
        P=6000lb: The applied load on the beam.
        L=14in: The length of the beam.
        δmax(delta)=0.25in: The maximum allowable deflection of the beam.
        E=30x106psi: The modulus of elasticity of the material.
        G=12x106psi: The shear modulus of the material.
        τmax(tau)=13600psi: The maximum allowable shear stress.
        σmax(sigma)=30000psi: The maximum allowable normal stress.
        """
        self.P = P
        self.L = L
        self.t_max = t_max
        self.s_max = s_max

    def evaluate_slack_true(self, X: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4 = X.unbind(-1)

        R = torch.sqrt(0.25 * (x2**2 + (x1 + x3) ** 2))
        M = self.P * (self.L + x2 / 2)
        J = 2 * math.sqrt(0.5) * x1 * x2 * (x2**2 / 12 + 0.25 * (x1 + x3) ** 2)
        t1 = self.P / (math.sqrt(2) * x1 * x2)
        t2 = M * R / J
        t = torch.sqrt(t1**2 + t1 * t2 * x2 / R + t2**2)
        s = 6 * self.P * self.L / (x4 * x3**2)
        P_c = 64746.022 * (1 - 0.0282346 * x3) * x3 * x4**3

        g1 = (t - self.t_max) / self.t_max
        g2 = (s - self.s_max) / self.s_max
        g3 = 1 / (5 - 0.125) * (x1 - x4)
        g4 = (self.P - P_c) / self.P

        return torch.stack([g1, g2, g3, g4], dim=-1)

    def get_constraints(self) -> List[Tuple[Callable, bool]]:
        constraints = [
            (lambda X: self.evaluate_slack_true(X)[:, 0], True),
            (lambda X: self.evaluate_slack_true(X)[:, 1], True),
            (lambda X: self.evaluate_slack_true(X)[:, 2], True),
            (lambda X: self.evaluate_slack_true(X)[:, 3], True)
        ]
        return constraints