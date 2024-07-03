from __future__ import annotations
from gpytorch.models import ExactGP
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.kernels import Kernel
from torch.optim.optimizer import Optimizer
from optimization.acquisition_function_factory import AcquisitionFunctionFactory
import torch
import warnings
import matplotlib.pyplot as plt
from botorch.optim import optimize_acqf


# TODO: IMPORTANT: new newly added data needs to be scaled with the same mean and std like the inital dataset!
# NOTE: optimization_loop(gp_model = self.gp_model, mll=self.mll, optimizer = self.optimizer, train_X = self.train_X, train_Y = self.train_Y, bounds = self.bounds_list, manual_input = manual_input, convergence_criterium = convergence_criterium, max_iterations = max_iterations)

# TODO: how is it implemented, that the target variables have minimum values? Can this be a stopping criteria in some way? That then a target is reached?
# TODO: how is the retraining implemented? How can I efficiently implement a retraining schedule for the model (e.g. after each new datapoint for the first 10 iterations, then after every thenth new datapoint?)
# TODO: can the implementation of optimize_acqf_list be helpful to simultaneously evaluate from a list of acquisition functions

# TODO: keep in mind to change the gp_model from the handed over base_model class!
# TODO: is minimization effectively handled?
# TODO: implementation of q? can this be helpful to tune?

# TODO: add dict support for the setup of the acq function, num_epochs, ...
# TODO: add support for dict or setup of acq function values such as num_restarts, raw_samples, ...

class OptimizerManager:
    pass

# class GPOptimizer():

#     def __init__(self, base_model: 'BaseGPModel', acq_func_type: str, is_maximization: bool = True) -> None:
#         self.base_model = base_model
#         self.is_maximization = is_maximization
#         self.acq_func_type = acq_func_type
#         self.acq_func = None
#         self.acq_func_plot_dict = {}
#         self.acq_func_values = []
#         self.next_observation_point = None
#         self.next_proposed_parameter_setting, _ = self.get_optimization_values(num_restarts=40, raw_samples=400)
    
#     # TODO: this function needs rebuilding
    
#     def optimization_loop(self, num_restarts: int, raw_samples: int, convergence_criteria: str = 'obj_func_pred_opt', manual_input: bool = True, max_iterations: int = 100):
#         #convergence of objective function AND prediction output OR of the acq_func_value?
        
#         for iteration in range(max_iterations):
#             candidate, acq_value = self.optimization_iteration(num_restarts=num_restarts, raw_samples=raw_samples)
        
#             if manual_input:
#                 print(f"Next suggested x-point: {candidate.item():.4f}")
#                 while True:
#                     try:
#                         next_y_value = float(input("Enter observation for given x: "))

#                         break 
#                     except ValueError:
#                         print("Invalid input. Please enter a valid floating-point number as an input")
#             else:
#                 raise ValueError(f"Automatic Input not supported yet in the optimization loop!")
            

#             next_y_value = self.convert_y_input_tensor(y_input=next_y_value)
            
#             #check if retraining is necessary - then set flag accordingly - flag is handed over in dataset add function of the model!
#             #where is the retraining added? This should be done after adding the next point - training function of the model can be used

#             self.base_model.add_point_to_dataset(new_X = candidate, new_Y = next_y_value)

#             self.base_model.train(num_epochs=100)

#             # TODO: do i need the refine still, if the model is already trained before?
#             # TODO: current guess: condition_ob_observation is not the correct way in handling the data here!
        
#     def optimization_iteration(self, observation:float, num_restarts=40, raw_samples=400):

#         self.add_new_observation_to_optimizer(observation=observation)
        
#         self.base_model.add_point_to_dataset(new_X = self.next_proposed_parameter_setting, new_Y = self.next_observation_point)

#         self.base_model.train(num_epochs=100)

#         candidate, acq_value = self.get_optimization_values(num_restarts=num_restarts, raw_samples=raw_samples)

#         self.next_proposed_parameter_setting = candidate
#         self.acq_func_values.append(acq_value.item())

        

#     def get_optimization_values(self, num_restarts=40, raw_samples=400):
        
#         if self.base_model.gp_model is None:
#             raise ValueError("No GP model set. Please run a train loop first!")

#         self.acq_func = AcquisitionFunctionFactory.create_acquisition_function(acq_function_type=self.acq_func_type, gp_model=self.base_model.gp_model, train_Y= self.base_model.train_Y, maximization=self.is_maximization)

#         candidate, acq_value = optimize_acqf(
#             acq_function=self.acq_func, 
#             bounds=self.base_model.bounds_list,
#             q=1,
#             num_restarts=num_restarts, 
#             raw_samples=raw_samples
#         )

#         self.acq_func_values.append(acq_value.item())

#         return candidate, acq_value

#     def add_new_observation_to_optimizer(self, observation: float):
#         observation = self.convert_y_input_tensor(observation)
#         self.next_observation_point = observation


#     def convert_y_input_tensor(self, y_input: float):
#         '''
#         Function to convert the manual y_input to a tensor in the correct shape!
#         '''
#         tensor_value = torch.tensor([[y_input]], dtype=torch.float64)
#         return tensor_value

#     def plot_acq_func(self, num_points: int = 100):
#         if self.acq_func is None:
#             raise ValueError("Acquisition function not set. Please run an optimization iteration first.")
        
#         bounds = self.base_model.bounds_list
#         lower_bounds = bounds[0].clone().detach()
#         upper_bounds = bounds[1].clone().detach()
        
#         # Create a grid for plotting
#         points = [torch.linspace(lower, upper, num_points) for lower, upper in zip(lower_bounds, upper_bounds)]
#         grid = torch.meshgrid(*points, indexing='ij')
#         grid_flat = torch.stack([g.reshape(-1) for g in grid], dim=-1)
        
#         # Reshape to ensure each design point is in the format (1 x d)
#         grid_flat = grid_flat.unsqueeze(1)  # Changes shape from N x d to N x 1 x d
        
#         total_points = num_points ** len(points)
#         if total_points > 10000:
#             warnings.warn(f"Warning: The acquisition function will evaluate {total_points} points, which may be computationally intensive.")

#         self.acq_func = AcquisitionFunctionFactory.create_acquisition_function(acq_function_type=self.acq_func_type, gp_model=self.base_model.gp_model, train_Y= self.base_model.train_Y, maximization=self.is_maximization)

#         with torch.no_grad():
#             acq_values = self.acq_func(grid_flat).squeeze(-1)

#         acq_values_reshaped = acq_values.view(*[num_points] * len(points))

#         plots = {}

#         for i in range(len(points)):
#             if len(points) == 1:
#                 dim_acq_values_avg = acq_values_reshaped
#             else:
#                 mean_dims = list(range(len(points)))
#                 mean_dims.remove(i)
#                 dim_acq_values_avg = acq_values_reshaped.mean(dim=tuple(mean_dims))

#             dim_points = points[i]

#             fig, ax = plt.subplots()
#             if self.acq_func_type == "LogExp_Improvement":
#                 dim_acq_values_avg = torch.exp(dim_acq_values_avg)
#             ax.plot(dim_points.numpy(), dim_acq_values_avg.numpy())
#             ax.set_xlabel(f"Dimension {i+1}")
#             ax.set_ylabel("Average Acquisition Value")
#             ax.set_title(f"Acquisition Function Averaged Over Other Dimensions - Dimension {i+1}")
#             plots[f"Dimension {i+1}"] = fig

#         self.acq_func_plot_dict = plots