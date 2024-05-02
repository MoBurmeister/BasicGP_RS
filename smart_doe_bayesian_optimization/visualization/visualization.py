from gpytorch.models import ExactGP
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import torch

# TODO: maybe later fix the "left out" dimensions to the something else than the mean? - could be the mode or a specific reference point
# TODO: Midpoints are for example also possibly feasible
# TODO: Potential future implementation of Partial Dependence Plots (PDPs) with sampled mean or other representative point

# TODO: IMPORTANT: Consider the scaling of the data!
# TODO: IMPORTANT: Consider the different dimensions in the data!
# TODO: add support for mulit output (y1, y2, ...)

class GPVisualizer:

    @staticmethod
    def visualize_model_pdp_with_uncertainty(gp_model: ExactGP, train_X: torch.tensor, train_Y: torch.tensor, scaling_dict: Dict, rescale_vis: bool = False, num_points=100):
        plots = {}
        input_dim = train_X.shape[1]

        # Compute means of all dimensions
        means = train_X.mean(0)

        # Fetch original bounds from the scaling_dict
        scaled_bounds = scaling_dict['inputs']['scaled_bounds']

        for dim in range(input_dim):
            # Create a grid for the dimension of interest
            min_val = scaled_bounds[0][dim].item()
            max_val = scaled_bounds[1][dim].item()

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

            # Rescale if required
            if rescale_vis:
                input_scale_info = scaling_dict['inputs']
                output_scale_info = scaling_dict['outputs']

                # Rescale inputs
                if input_scale_info['method'] == 'normalize':
                    min_input_val, max_input_val = input_scale_info['params']['min'][dim].numpy(), input_scale_info['params']['max'][dim].numpy()
                    x_range = x_range * (max_input_val - min_input_val) + min_input_val
                    observed_data = train_X[:, dim].numpy() * (max_input_val - min_input_val) + min_input_val
                elif input_scale_info['method'] == 'standardize':
                    mean_input_val, std_input_val = input_scale_info['params']['mean'][dim].numpy(), input_scale_info['params']['std'][dim].numpy()
                    x_range = x_range * std_input_val + mean_input_val
                    observed_data = train_X[:, dim].numpy() * std_input_val + mean_input_val

                # Rescale outputs
                if output_scale_info['method'] == 'normalize':
                    min_output_val, max_output_val = output_scale_info['params']['min'].numpy(), output_scale_info['params']['max'].numpy()
                    mean = mean * (max_output_val - min_output_val) + min_output_val
                    lower = lower * (max_output_val - min_output_val) + min_output_val
                    upper = upper * (max_output_val - min_output_val) + min_output_val
                    train_Y_rescaled = train_Y.numpy() * (max_output_val - min_output_val) + min_output_val
                elif output_scale_info['method'] == 'standardize':
                    mean_output_val, std_output_val = output_scale_info['params']['mean'].numpy(), output_scale_info['params']['std'].numpy()
                    mean = mean * std_output_val + mean_output_val
                    lower = lower * std_output_val + mean_output_val
                    upper = upper * std_output_val + mean_output_val
                    train_Y_rescaled = train_Y.numpy() * std_output_val + mean_output_val

            # Plotting
            fig, ax = plt.subplots()
            ax.plot(x_range, mean, label='Predictive Mean')
            ax.fill_between(x_range, lower, upper, alpha=0.5, label='95% Confidence Interval')
            ax.scatter(observed_data, train_Y_rescaled, c='red', label='Observations')
            ax.legend()
            ax.set_xlabel(f'Input dimension {dim+1}')
            ax.set_ylabel('Output')
            ax.set_title(f'GP Model Visualization for Input Dimension {dim+1}')

            plt.show()

            # Store plot
            plots[f'dim_{dim+1}'] = fig

        return plots