from __future__ import annotations
from models.gp_model import BaseModel
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
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
import numpy as np

class BayesianOptimizer:

    def __init__(
        self,
        multiobjective_model: BaseModel,
        parameter_constraints_equality: Optional[Callable] = None,
        parameter_constraints_inequality: Optional[Callable] = None,
        parameter_constraints_nonlinear_inequality: Optional[Callable] = None,
        output_constraints: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        reference_point: Optional[torch.Tensor] = None, 
        save_file_name: Optional[str] = None
    ) -> None:
        self.multiobjective_model = multiobjective_model
        if reference_point is not None:
            self.reference_point = reference_point
            self.reference_point_handed_over = True
            print(f"Reference Point handed over: {reference_point}")
        else:
            self.reference_point = self.calculate_reference_point()
            self.reference_point_handed_over = False

        # No implementation of constraints yet. Will cause way longer runtime and not necessary yet
        self.parameter_constraints_equality = parameter_constraints_equality
        self.parameter_constraints_inequality = parameter_constraints_inequality
        self.parameter_constraints_nonlinear_inequality = parameter_constraints_nonlinear_inequality    

        self.output_constraints = output_constraints
        self.next_input_setting = None
        self.next_observation = None
        self.external_input = multiobjective_model.dataset_manager.external_input
        self.save_file_name = save_file_name
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

        # Compare element-wise and update to the worse value (lower value) for each dimension
        updated_ref_point = torch.min(self.reference_point, new_ref_point)

        if not torch.equal(updated_ref_point, self.reference_point):
            self.reference_point = updated_ref_point
            print(f"Reference Point updated to: {self.reference_point}")
        else:
            print("New Reference Point is not worse in any dimension. No update performed.")

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

        if self.multiobjective_model.dataset_manager.initial_dataset.input_data.shape[0] == 0:
            prune_baseline_check = False
            #print(f"Initial dataset is empty. Prune baseline set to False.")
        else:
            prune_baseline_check = True
            #print(f"Initial dataset is not empty. Prune baseline set to True.")

        acq_function = qLogNoisyExpectedHypervolumeImprovement(model=self.multiobjective_model.gp_model, 
                                                            ref_point=self.reference_point, 
                                                            X_baseline=self.multiobjective_model.dataset_manager.initial_dataset.input_data,
                                                            constraints=self.output_constraints, 
                                                            prune_baseline=prune_baseline_check)
        
        # TODO: what about num_restarts and raw_samples?
        # here implementation of input constraints but not supported yet!
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
            
            self.get_next_manual_observation()

            #get next input here manually, ckeck that it is negated according to max flags etc.
        else:
            print("External input is set to False. Target Observation is not provided manually and instead via function internally.")
            self.get_next_observation()

        #only adjusted next_observation (based on maximization flags) will be added to dataset!        
        self.multiobjective_model.dataset_manager.add_point_to_initial_dataset(point=(self.next_input_setting, self.next_observation))
        self.multiobjective_model.reinitialize_model(current_iteration = iteration_num)
        

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
    
    def get_next_manual_observation(self):
        
        next_observation = []

        # Gather the number of outputs (objectives) from the model's dataset manager
        num_outputs = self.multiobjective_model.dataset_manager.output_dim

        # Gather maximization flags
        maximization_flags = self.multiobjective_model.dataset_manager.maximization_flags

        # Placeholder for output parameter names
        output_parameter_names = self.multiobjective_model.dataset_manager.output_parameter_name

        print("Please provide the output values for the given input manually.")

        # Loop to get the output values for each objective from the user
        output_values = []
        for j in range(num_outputs):
            while True:
                try:
                    output_value = float(input(f"Enter the output value for the objective '{output_parameter_names[j]}' for this input point: "))
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")
            output_values.append(output_value)
        
        next_observation.append(output_values)

        # Convert the list to a torch tensor and reshape it to [1, num_outputs]
        next_observation = torch.tensor(output_values).view(1, num_outputs)

        # Negate the values in next_observation for dimensions where the maximization flag is False
        for i, flag in enumerate(maximization_flags):
            if not flag:
                next_observation[:, i] = -next_observation[:, i]

        self.next_observation = next_observation

        print(f"Next manual observation: {self.next_observation}")

    '''
    Hypervolume improvement quantifies how much the hypervolume would increase if a new point (or set of points) 
    were added to the current Pareto front.
    '''
    def optimization_loop(self, use_stopping_criterion: bool = False, num_max_iterations: int = 10, num_min_iterations: int = 40):

        if num_max_iterations < num_min_iterations:
            raise ValueError("The number of maximum iterations must be greater than or equal to the number of minimum iterations.")

        #calculate and add initial hypervolume:
        initial_hypervolume = self.calculate_hypervolume()
        self.optimization_loop_data_dict[0] = {"hypervolume": initial_hypervolume}
        print(f"Inital Hypervolume: {initial_hypervolume}")

        #initiating stopping criterion classes
        #Note: set minimize to false, when considered measurement is maximized (e.g. hypervolume)
        if use_stopping_criterion:
            stopping_criterion_hypervolume = Extended_ExpMAStoppingCriterion(minimize=False, n_window=35, eta=1.0, rel_tol=0.001)

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
        
            print(f"Iteration {iteration + 1} of max. {num_max_iterations} iterations completed. It took {iteration_duration:.2f} seconds.")

            #Modulo to potentially adjust computationally expensive calculation of the hypervolume
            if iteration % 1 == 0:
                hypervolume = self.calculate_hypervolume()
                self.optimization_loop_data_dict[iteration+1]["hypervolume"] = hypervolume
                print(f"Final Hypervolume: {hypervolume}")

            #Check to update reference point every 10 iterations. Potentially adjust the number 10. Also just update ref point when it is not handed over
            if iteration % 4 == 0 and not self.reference_point_handed_over:
                self.update_reference_point()
                self.optimization_loop_data_dict[iteration+1]["reference_point"] = self.reference_point

            if use_stopping_criterion:
                if self.stopping_criterion(num_iteration = iteration, sc_hypervolume = stopping_criterion_hypervolume) and iteration >= num_min_iterations:
                    print(f"Stopping criterion reached after {iteration + 1} iterations with min iterations being {num_min_iterations} and max iterations being {num_max_iterations}. Breaking the optimization loop.")
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
        
        print(50*"#")

        # Print the total time taken
        print(f"Total time taken for optimization: {total_time:.2f} seconds. Deviations in the summed iteration times may be possible due to additional calculations outside the iterations (e.g., Hypervolume).")
        
        self.visualize_expected_hypervolume_development()
        self.visualize_parallel_coordinates_plot()
        self.visualize_pareto_front()

        # Get the current date and format it
        current_date_time = datetime.now().strftime("%Y%m%d_%H%M")

        # Create a folder name based on the current date and time
        folder_name = f"{current_date_time}_BOMOGP_TL_Opt_{self.save_file_name}"
        
        folder_path = os.path.join("smart_doe_bayesian_optimization", "data_export", "multi_singletaskgp_data_export")

        # Export everything via function
        export_everything(multiobjective_model=self.multiobjective_model, optimization_dict=self.optimization_loop_data_dict, results_dict=self.results_dict, fig_list=self.export_figures, folder_path=folder_path, folder_name=folder_name, file_format="xlsx")
        print(f"Optimization data exported to folder in path: {folder_path}")
        print(50*"#")

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
    