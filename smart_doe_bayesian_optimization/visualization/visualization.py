from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.gp_model import BaseModel
import itertools



class GP_Visualizer:

    @staticmethod
    def visualize_pareto_front_scatter(multiobjective_model: BaseModel, results_dict: Dict):
        '''
        Function to visualize the pareto front of a multiobjective model with scatter plots for each pair of objectives.
        Important: Backtransformation should be applied here!
        '''
        num_output_dim = multiobjective_model.dataset_manager.initial_dataset.output_dim

        number_of_scatter_plots = num_output_dim * (num_output_dim - 1) // 2

        pareto_points = results_dict["pareto_points"]

        print(50*"-")
        print(f"Pareto Points: {pareto_points}")
        #If a flag is False, it indicates the corresponding output dimension should be minimized!
        #The method will negate the values in that output dimension.
        maximization_flags = multiobjective_model.dataset_manager.initial_dataset.maximization_flags

        print(f"maximization Flags: {maximization_flags}")

        # Create a copy to avoid modifying the original pareto_points
        reverted_pareto_points = pareto_points.clone()

        for i, flag in enumerate(maximization_flags):
            if not flag:
                reverted_pareto_points[:, i] = -reverted_pareto_points[:, i]

        print(f"Reverted Pareto Points: {reverted_pareto_points}")

        # Create pairwise scatter plots
        pairs = list(itertools.combinations(range(num_output_dim), 2))
        
        fig, axes = plt.subplots(len(pairs) // 2 + len(pairs) % 2, 2, figsize=(15, 5 * (len(pairs) // 2 + len(pairs) % 2)))
        axes = axes.flatten()

        for idx, (i, j) in enumerate(pairs):
            ax = axes[idx]
            ax.scatter(reverted_pareto_points[:, i], reverted_pareto_points[:, j])
            ax.set_xlabel(f'Objective {i + 1}')
            ax.set_ylabel(f'Objective {j + 1}')
            ax.set_title(f'Objective {i + 1} vs Objective {j + 1}')

        # Hide any unused subplots
        for idx in range(len(pairs), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_hypervolume_improvement(optimization_data_dict: Dict):
        """
        Visualizes the hypervolume improvements over iterations.

        Args:
            optimization_data_dict (Dict): The dictionary containing optimization data.
        """
        iterations = []
        acq_values = []

        for iteration, data in optimization_data_dict.items():
            if "acq_value" in data:
                iterations.append(iteration)
                acq_values.append(data["acq_value"])

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, acq_values, marker='o', linestyle='-', color='b')
        plt.xlabel('Iteration')
        plt.ylabel('Expected Hypervolume Improvement')
        plt.title('Hypervolume Improvement over Iterations')
        plt.grid(True)
        plt.show()
        

        



# class GPVisualizer:

#     @staticmethod
#     def visualize_model_pdp_with_uncertainty(gp_model: ExactGP, train_X: torch.tensor, train_Y: torch.tensor, bounds_list: torch.Tensor, num_points: int =100):
#         plots = {}
#         input_dim = train_X.shape[1]

#         # Compute means of all dimensions
#         means = train_X.mean(0)

#         for dim in range(input_dim):
#             # Create a grid for the dimension of interest
#             min_val = bounds_list[0, dim]
#             max_val = bounds_list[1, dim]

#             # Create a grid for the dimension of interest
#             x_range = torch.linspace(min_val, max_val, num_points)
#             grid = means.repeat(num_points, 1)
#             grid[:, dim] = x_range

#             # Get model predictions
#             gp_model.eval()
#             with torch.no_grad():
#                 posterior = gp_model.posterior(grid)
#                 mean = posterior.mean.detach().numpy()
#                 lower, upper = posterior.mvn.confidence_region()

#             # Plotting
#             fig, ax = plt.subplots()
#             ax.plot(x_range, mean, label='Predictive Mean')
#             ax.fill_between(x_range, lower, upper, alpha=0.5, label='95% Confidence Interval')
#             # TODO: potentially here: .cpu() at the end of each tensor?
#             ax.scatter(train_X[:, dim].numpy(), train_Y.numpy(), c='red', label='Observations')
#             ax.legend()
#             ax.set_xlabel(f'Input dimension {dim+1}')
#             ax.set_ylabel('Output')
#             ax.set_title(f'GP Model Visualization for Input Dimension {dim+1}')

#             # Store plot
#             plots[f'dim_{dim+1}'] = {
#                 'plot': fig,
#                 'range': (min_val.item(), max_val.item()),
#                 'mean': means[dim].item()
#             }

#         return plots