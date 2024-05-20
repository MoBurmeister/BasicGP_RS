from gpytorch.models import ExactGP
from typing import Dict
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive backend 'Agg'
import matplotlib.pyplot as plt
import numpy as np
import torch

# TODO: maybe later fix the "left out" dimensions to the something else than the mean? - could be the mode or a specific reference point
# TODO: Potential future implementation of Partial Dependence Plots (PDPs) with sampled mean or other representative point

# TODO: add support for mulit output (y1, y2, ...)

class GPVisualizer:

    @staticmethod
    def visualize_model_pdp_with_uncertainty(gp_model: ExactGP, train_X: torch.tensor, train_Y: torch.tensor, bounds_list: torch.Tensor, num_points: int =100):
        plots = {}
        input_dim = train_X.shape[1]

        # Compute means of all dimensions
        means = train_X.mean(0)

        for dim in range(input_dim):
            # Create a grid for the dimension of interest
            min_val = bounds_list[0, dim]
            max_val = bounds_list[1, dim]

            # Create a grid for the dimension of interest
            x_range = torch.linspace(min_val, max_val, num_points)
            grid = means.repeat(num_points, 1)
            grid[:, dim] = x_range

            # Get model predictions
            gp_model.eval()
            with torch.no_grad():
                posterior = gp_model.posterior(grid)
                mean = posterior.mean.detach().numpy()
                lower, upper = posterior.mvn.confidence_region()

            # Plotting
            fig, ax = plt.subplots()
            ax.plot(x_range, mean, label='Predictive Mean')
            ax.fill_between(x_range, lower, upper, alpha=0.5, label='95% Confidence Interval')
            # TODO: potentially here: .cpu() at the end of each tensor?
            ax.scatter(train_X[:, dim].numpy(), train_Y.numpy(), c='red', label='Observations')
            ax.legend()
            ax.set_xlabel(f'Input dimension {dim+1}')
            ax.set_ylabel('Output')
            ax.set_title(f'GP Model Visualization for Input Dimension {dim+1}')

            # Store plot
            plots[f'dim_{dim+1}'] = {
                'plot': fig,
                'range': (min_val.item(), max_val.item()),
                'mean': means[dim].item()
            }

        return plots