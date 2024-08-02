from __future__ import annotations
from models.gp_model import BaseModel
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
import torch
from botorch.optim import optimize_acqf
from typing import Callable, Optional, List
from visualization.visualization import GP_Visualizer   
import time
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated 
from data_export.data_export import export_everything   
from datetime import datetime  
import os
from optimization.stopping_criterion import Extended_ExpMAStoppingCriterion


# TODO: Implementation of parameter_constraints

class BayesianOptimizer:

    def __init__(
        self,
        multiobjective_model: BaseModel,
        parameter_constraints_equality: Optional[Callable] = None,
        parameter_constraints_inequality: Optional[Callable] = None,
        parameter_constraints_nonlinear_inequality: Optional[Callable] = None,
        output_constraints: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        reference_point: Optional[torch.Tensor] = None,
        external_input: bool = False
    ) -> None:
        self.multiobjective_model = multiobjective_model
        if reference_point is not None:
            self.reference_point = reference_point
            print(f"Reference Point handed over: {reference_point}")
        else:
            self.reference_point = self.calculate_reference_point()

        # TODO: Check for the correct setup of the constraints? Necessary? is already checked in botorch implementation?

        self.parameter_constraints_equality = parameter_constraints_equality
        self.parameter_constraints_inequality = parameter_constraints_inequality
        self.parameter_constraints_nonlinear_inequality = parameter_constraints_nonlinear_inequality    
        self.output_constraints = output_constraints
        self.next_input_setting = None
        self.next_observation = None
        self.external_input = external_input
        self.gp_visualizer = GP_Visualizer()
        #stores: hypervolume, acq_value, iteration_duration, stopping criterion ma_values for hypervolume and acq_value
        self.optimization_loop_data_dict = {}
        #stores mainly pareto points right now:
        self.results_dict = {}
        self.hypervolume_calculator = Hypervolume(ref_point=self.reference_point)
        self.export_figures = []

        print(50*"-")   
        print(f"Bayesian Optimizer initialized.")

    def calculate_reference_point(self):
            '''
            The reference point is necessary as a base point for the hypervolume calculation.
            Reference point will be calculated as the minimum (worst point) based on the current optimization dataset 
            (no historic datasets considered)
            It is based on all the objectives; minimum (worst) since everything is maximized

            all outcomes must be greater than the corresponding ref_point value!
            '''
            # Get the minimum values across all the objectives (output dimensions)
            min_values = torch.min(self.multiobjective_model.dataset_manager.initial_dataset.output_data, dim=0).values

            # Convert min_values to torch.float64 to ensure the dtype is correct
            min_values = min_values.to(torch.float64)

            worsening_percentage = 5

            # Calculate the worsening factor based on whether the min_value is positive or negative
            worsening_factor = torch.ones_like(min_values)
            worsening_factor[min_values >= 0] = 1 - (worsening_percentage / 100)  # for non-negative values, reduce by percentage
            worsening_factor[min_values < 0] = 1 + (worsening_percentage / 100)   # for negative values, increase by percentage

            # Set the reference point based on the calculated worsening factors
            reference_point = min_values * worsening_factor

            print(f"Reference Point calculated: {reference_point}")

            return reference_point
    
    def update_reference_point(self):

        new_ref_point = self.calculate_reference_point()

        # Check if any element of the new reference point is "worse" than the old one
        if torch.any(new_ref_point < self.reference_point):
            self.reference_point = new_ref_point
            print(f"Reference Point updated to: {self.reference_point}")
        else:
            print("New Reference Point is not worse than the current one. No update performed.")

    def optimization_iteration(self, iteration_num: int):
        #in the first stage I will only incorporate the qnehvi acquisition function
        '''
        reason for qNEHVI: 
        qNEHVI utilizes some nice tricks that makes it much more scalable than
        qEHVI wrt the batch size (parallelism), q.  Empirically the performance is
        at least as good in the noiseless case and better than any other AF in the
        noisy case. For this reason we’d recommend qNEHVI as the default multi
        objective BO algorithm to use (and we default to it in Ax).
        '''
        '''
        Since botorch assumes a maximization of all objectives, we seek to find the Pareto frontier, 
        the set of optimal trade-offs where improving one metric means deteriorating another.
        '''

        self.validate_output_constraints()

        # TODO: Should I input more arguments?
        acq_function = qNoisyExpectedHypervolumeImprovement(model=self.multiobjective_model.gp_model, 
                                                            ref_point=self.reference_point, 
                                                            X_baseline=self.multiobjective_model.dataset_manager.initial_dataset.input_data,
                                                            constraints=self.output_constraints, 
                                                            prune_baseline=True)
        
        # TODO: what about num_restarts and raw_samples?
        # here implementation of input constraints
        candidate, acq_value = optimize_acqf(
            acq_function=acq_function,
            bounds=self.multiobjective_model.dataset_manager.initial_dataset.bounds_list,
            q=1,
            num_restarts=40,
            raw_samples=512,
            inequality_constraints=self.parameter_constraints_inequality,
            equality_constraints=self.parameter_constraints_equality,
            nonlinear_inequality_constraints=self.parameter_constraints_nonlinear_inequality
        )

        # error potential when acq_value is a tensor with multiple values
        self.optimization_loop_data_dict[iteration_num + 1]["acq_value"] = acq_value.item()

        print(f"Next suggested input-point: {candidate.tolist()} with acquisition value: {acq_value.item()}")

        self.next_input_setting = candidate

        if self.external_input:
            print("External input is set to True. Target Observation is provided manually.")
            raise ValueError("External input not supported yet!")
        else:
            print("External input is set to False. Target Observation is not provided manually and instead via function internally.")
            self.get_next_observation()
        
        self.multiobjective_model.dataset_manager.add_point_to_initial_dataset(point=(self.next_input_setting, self.next_observation))
        self.multiobjective_model.reinitialize_model()
        

    def validate_output_constraints(self):
        '''
        Validate if the output constraints are correctly setup.
        '''
        if self.output_constraints is not None:
            if not isinstance(self.output_constraints, list):
                raise ValueError("output_constraints should be a list of callables")
            
            for constraint in self.output_constraints:
                if not callable(constraint):
                    raise ValueError("Each constraint should be a callable function")
                
                # Check the constraint's output dimensions
                #test_tensor = torch.rand(1, 1, 1, 1)  # Example tensor with shape (sample_shape, batch_shape, q, m)
                ##constraint_output = constraint(test_tensor)
                #if constraint_output.shape != test_tensor.shape[:-1]:
                #    raise ValueError("Each constraint should return a Tensor of shape (sample_shape x batch-shape x q)")
            

    def get_next_observation(self):
        #I need to treat the results from the dataset_func according to the minimization flags here:
        #unnecessary expected output dimension here captured as _
        next_observation, _ = self.multiobjective_model.dataset_manager.dataset_func(self.next_input_setting)

        maximization_flags = self.multiobjective_model.dataset_manager.maximization_flags

        # Negate the values in next_observation for dimensions where the maximization flag is False
        for i, flag in enumerate(maximization_flags):
            if not flag:
                next_observation[:, i] = -next_observation[:, i]

        self.next_observation = next_observation

        print(f"Next observation: {self.next_observation}")  

    '''
    Hypervolume improvement quantifies how much the hypervolume would increase if a new point (or set of points) 
    were added to the current Pareto front.
    '''
    def optimization_loop(self, use_stopping_criterion: bool = False, num_max_iterations: int = 10):

        #calculate and add initial hypervolume:
        initial_hypervolume = self.calculate_hypervolume()
        self.optimization_loop_data_dict[0] = {"hypervolume": initial_hypervolume}
        print(f"Inital Hypervolume: {initial_hypervolume}")

        #initiating stopping criterion classes
        #set minimize to false, if considered measurement is maximized (e.g. hypervolume)
        if use_stopping_criterion:
            stopping_criterion_hypervolume = Extended_ExpMAStoppingCriterion(minimize=False, n_window=15, eta=1.0, rel_tol=1e-6)

        start_time = time.time()

        #this loop works on a stopping criterion, see above
        for iteration in range(num_max_iterations):
            iteration_start_time = time.time()

            self.optimization_loop_data_dict[iteration + 1] = {}

            self.optimization_iteration(iteration_num = iteration)
            #after this the next best point is found, sampled and also added to the dataset!
            iteration_end_time = time.time()
            iteration_duration = iteration_end_time - iteration_start_time

            self.optimization_loop_data_dict[iteration+1]["iteration_duration"] = iteration_duration
        
            print(f"Iteration {iteration + 1} of {num_max_iterations} iterations completed. It took {iteration_duration:.2f} seconds.")

            #Modulo to potentially adjust computationally expensive calculation of the hypervolume
            if iteration % 1 == 0:
                hypervolume = self.calculate_hypervolume()
                self.optimization_loop_data_dict[iteration+1]["hypervolume"] = hypervolume
                print(f"Final Hypervolume: {hypervolume}")

            #Check to update reference point every 10 iterations
            if iteration % 10 == 0:
                self.update_reference_point()

            if use_stopping_criterion:
                if self.stopping_criterion(num_iteration = iteration, sc_hypervolume = stopping_criterion_hypervolume):
                    print(f"Stopping criterion reached after {iteration + 1} iterations. Breaking the optimization loop.")
                    break
            
            #add moving average values, but these exists only after window size of ma is reached, therefore if else check - prob not b.p.
            if use_stopping_criterion and stopping_criterion_hypervolume.ma_values:
                self.optimization_loop_data_dict[iteration+1]["hypervolume_ma_value"] = stopping_criterion_hypervolume.ma_values[-1]
                self.optimization_loop_data_dict[iteration+1]["hypervolume_ma_rel_value"] = stopping_criterion_hypervolume.rel_values[-1]
            else:
                self.optimization_loop_data_dict[iteration+1]["hypervolume_ma_value"] = None
                self.optimization_loop_data_dict[iteration+1]["hypervolume_ma_rel_value"] = None       

            print(50*"*")

        # End measuring time
        end_time = time.time()
        
        # Calculate the total time taken
        total_time = end_time - start_time
        
        # Print the total time taken
        print(f"Total time taken for optimization: {total_time:.2f} seconds. Deviations in the summed iteration times may be possible due to additional calculations outside the iterations (e.g., Hypervolume).")
        
        self.visualize_expected_hypervolume_development()
        self.visualize_parallel_coordinates_plot()
        self.visualize_pareto_front()

        #print(f"Optimization data dictionary: {self.optimization_loop_data_dict}")

        #print(f"Results dictionary: {self.results_dict}")

        # Get the current date and format it
        current_date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create a folder name based on the current date and time
        folder_name = f"{current_date_time}_BOMOGP_TL_Optimization_Results"
        
        folder_path = os.path.join("smart_doe_bayesian_optimization", "data_export", "multi_singletaskgp_data_export")

        # Export everything via function
        export_everything(multiobjective_model=self.multiobjective_model, optimization_dict=self.optimization_loop_data_dict, results_dict=self.results_dict, fig_list=self.export_figures, folder_path=folder_path, folder_name=folder_name, file_format="xlsx")
        print(f"Optimization data exported to folder in path: {folder_path}")

    def calculate_hypervolume(self):

        self.calculate_pareto_points()

        hypervolume = self.hypervolume_calculator.compute(pareto_Y=self.results_dict["pareto_points"])

        return hypervolume
    
    def calculate_pareto_points(self):

        output_data = self.multiobjective_model.dataset_manager.initial_dataset.output_data

        pareto_boolean_tensor = is_non_dominated(Y=output_data)

        pareto_points = output_data[pareto_boolean_tensor]

        self.results_dict["pareto_points"] = pareto_points

    def visualize_pareto_front(self):
        fig = self.gp_visualizer.visualize_pareto_front_scatter(self.multiobjective_model, self.results_dict)
        self.export_figures.append(fig)

    def visualize_expected_hypervolume_development(self):
        fig = self.gp_visualizer.visualize_hypervolume_improvement(self.optimization_loop_data_dict)
        self.export_figures.append(fig)  

    def visualize_parallel_coordinates_plot(self):
        fig = self.gp_visualizer.visualize_parallel_coordinates_plot(self.multiobjective_model, self.results_dict)
        self.export_figures.append(fig)

    def stopping_criterion(self, num_iteration: int, sc_hypervolume: Extended_ExpMAStoppingCriterion):

        if sc_hypervolume.evaluate(torch.tensor(self.optimization_loop_data_dict[num_iteration+1]["hypervolume"])):
            return True

        return False
    



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