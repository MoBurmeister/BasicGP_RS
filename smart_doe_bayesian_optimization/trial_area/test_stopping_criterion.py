import torch
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod

class StoppingCriterion(ABC):
    r"""Base class for evaluating optimization convergence."""
    @abstractmethod
    def evaluate(self, fvals: torch.Tensor) -> bool:
        r"""Evaluate the stopping criterion."""
        pass  # pragma: no cover

    def __call__(self, fvals: torch.Tensor) -> bool:
        return self.evaluate(fvals)

class Extended_ExpMAStoppingCriterion(StoppingCriterion):
    r"""Exponential moving average stopping criterion."""
    def __init__(
        self,
        maxiter: int = 10000,
        minimize: bool = True,
        n_window: int = 10,
        eta: float = 1.0,
        rel_tol: float = 1e-5,
    ) -> None:
        self.maxiter = maxiter
        self.minimize = minimize
        self.n_window = n_window
        self.rel_tol = rel_tol
        self.iter = 0
        weights = torch.exp(torch.linspace(-eta, 0, self.n_window))
        self.weights = weights / weights.sum()
        self._prev_fvals = None
        self.ma_values = []  # List to store moving average values
        self.rel_values = [] # List to store relative decrease values

    def evaluate(self, fvals: torch.Tensor) -> bool:
        self.iter += 1
        if self.iter == self.maxiter:
            return True

        if self._prev_fvals is None:
            self._prev_fvals = fvals.unsqueeze(0)
        else:
            self._prev_fvals = torch.cat(
                [self._prev_fvals[-self.n_window :], fvals.unsqueeze(0)]
            )

        if self._prev_fvals.size(0) < self.n_window + 1:
            return False

        weights = self.weights
        weights = weights.to(fvals)
        if self._prev_fvals.ndim > 1:
            weights = weights.unsqueeze(-1)

        prev_ma = (self._prev_fvals[:-1] * weights).sum(dim=0)
        ma = (self._prev_fvals[1:] * weights).sum(dim=0)

        # Save the current moving average value
        self.ma_values.append(ma.item() if ma.numel() == 1 else ma.tolist())

        rel_delta = (prev_ma - ma) / prev_ma.abs()

        # Save the current relative decrease value
        self.rel_values.append(rel_delta.item() if rel_delta.numel() == 1 else rel_delta.tolist())

        if not self.minimize:
            rel_delta = -rel_delta
        if torch.max(rel_delta) < self.rel_tol:
            return True

        return False

# Hypervolume values
hypervolume_values_1 = [
    78.20302534, 79.27067438, 95.0377798, 96.10265378, 96.19252966, 113.021242, 
    118.9021048, 119.6540981, 123.9890214, 131.760898, 133.8406168, 135.2902002, 
    137.6030449, 140.5121012, 142.1357538, 148.3076023, 155.0023374, 163.7662378, 
    173.2990699, 178.9251856, 188.6378987, 192.4670457, 193.3100978, 195.4721886, 
    196.9709892, 199.7017816, 200.6711914, 203.707627, 206.1008496, 218.8098468, 
    224.8602815, 227.0169357, 227.2164555, 229.3019545, 230.4160789, 230.4160789, 
    235.0297959, 236.6476428, 236.9265723, 237.8800784, 239.5768208, 240.0477522, 
    240.3150136, 240.3150136, 240.73367, 241.0472547, 241.3393237, 241.3393237, 
    241.4543104, 241.6514878, 241.6514878, 241.8192475, 241.9744032, 241.9744032, 
    241.9744032, 241.9986031, 241.9986031, 242.1818798, 242.3413278, 242.4175092, 
    242.4642053, 242.5228322, 242.5382299, 242.5673081, 242.6554135, 242.6910812, 
    242.7602487, 242.7619234, 242.7619234, 242.9257908, 242.9257908, 242.9486849
]

#initial TL
hypervolume_values_2 = [
    56.62637337, 96.80611153, 97.55307653, 102.2476494, 197.2511563,
    197.2511563, 197.5159306, 197.5159306, 197.5159306, 200.2627582,
    208.7740191, 208.7740191, 208.7740191, 208.7740191, 208.7740191,
    211.8633279, 211.8633279, 211.8633279, 211.8633279, 211.8633279,
    217.6433341, 217.6433341, 219.7237337, 219.7237337, 219.7237337,
    223.3293981, 223.3293981, 223.3293981, 223.3293981, 223.3293981,
    223.3293981, 223.3293981, 223.3293981, 223.3293981, 223.7313743,
    223.7313743, 223.7313743, 223.7313743, 223.7313743, 223.7313743,
    224.8606131, 224.8606131, 224.8606131, 224.8606131, 224.8606131,
    224.8606131, 225.7246328, 225.7246328, 227.9371992, 227.9371992,
    227.9371992, 227.9371992, 227.9371992, 227.9371992, 227.9371992,
    227.9371992, 227.9371992, 229.6869294, 229.6869294, 229.6869294,
    229.6869294
]

#finetune TL
hypervolume_values_3 = [
    58.74751154, 152.8819846, 215.8157019, 215.8157019, 215.8157019,
    215.8179406, 219.9399076, 220.7452063, 229.1944159, 229.1944159,
    229.1944159, 229.1944159, 229.1944159, 229.4288903, 229.4288903,
    229.4288903, 229.4288903, 229.4288903, 229.4288903, 231.4287931,
    231.4287931, 231.4287931, 231.4505993, 231.4505993, 231.4505993,
    231.4505993, 235.3826754, 235.3826754, 235.3826754, 235.3847918,
    235.3847918, 235.3847918, 235.3847918, 236.4468353, 236.4468353,
    236.4468353, 236.4468353, 236.4468353, 236.4468353, 236.4468353,
    236.4468353, 236.4468353, 236.4476631, 236.774437
]

#multitask TL:
hypervolume_values_4 = [
    139.7206129, 148.7001721, 198.5325819, 208.8576315, 221.1748188,
    230.5647744, 232.3384599, 233.1878983, 233.915176, 234.9688579,
    236.1783225, 236.691517, 236.691517, 237.155634, 237.322622,
    237.4799827, 237.4799827, 237.4799827, 237.685344, 237.9734352,
    237.9734352, 237.9734352, 237.9867356, 238.1697276, 238.1939611,
    238.3040663, 238.533098, 238.5818285, 238.8409183, 238.9449365,
    239.3608058, 239.3608058, 239.3888094, 239.476322, 239.476322,
    239.6201203, 239.62799, 239.6612973, 239.6612973, 239.7758726,
    239.8092682, 239.8092682, 239.8092682, 239.8092682, 239.8092682,
    239.8092682, 239.8662274, 239.8662274, 239.9037375, 239.9037375,
    239.9037375, 239.9037375, 239.9037375, 239.9037375, 239.9037375,
    239.9037375, 239.9659813, 239.9932123, 239.9978166, 240.0501,
    240.0501, 240.0557575, 240.0557575, 240.0804984, 240.0804984,
    240.133812, 240.159913, 240.159913, 240.159913, 240.159913,
    240.159913
]



# Convert hypervolume values to torch tensor
fvals = torch.tensor(hypervolume_values_4)

# Initialize the stopping criterion
criterion = Extended_ExpMAStoppingCriterion(
    maxiter=1000,  # or some other large number
    minimize=False,  # Set to False as we are maximizing hypervolume
    n_window=15,  # You can adjust the window size
    eta=1.0,  # You can adjust the decay factor
    rel_tol=0.001  # You can adjust the relative tolerance
)

# Test the stopping criterion with the hypervolume values
for i, hv in enumerate(fvals):
    should_stop = criterion(hv)
    print(f"Iteration {i + 1}, Hypervolume: {hv.item()}, Should Stop: {should_stop}")
    if should_stop:
        print("Stopping criterion met.")
        break

# Optionally, you can plot the moving averages and relative decreases
plt.figure(figsize=(12, 6))
plt.plot(fvals, label='Moving Average')
plt.title('Exponential Moving Average')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.legend()


plt.tight_layout()
plt.show()
