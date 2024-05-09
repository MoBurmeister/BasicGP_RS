from __future__ import annotations
from gpytorch.models import ExactGP
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.kernels import Kernel
from torch.optim.optimizer import Optimizer
from optimization.acquisition_function_factory import AcquisitionFunctionFactory
import torch
from botorch.optim import optimize_acqf


# TODO: IMPORTANT: new newly added data needs to be scaled with the same mean and std like the inital dataset!
# NOTE: optimization_loop(gp_model = self.gp_model, mll=self.mll, optimizer = self.optimizer, train_X = self.train_X, train_Y = self.train_Y, bounds = self.bounds_list, manual_input = manual_input, convergence_criterium = convergence_criterium, max_iterations = max_iterations)

# TODO: how is it implemented, that the target variables have minimum values? Can this be a stopping criteria in some way? That then a target is reached?
# TODO: how is the retraining implemented? How can I efficiently implement a retraining schedule for the model (e.g. after each new datapoint for the first 10 iterations, then after every thenth new datapoint?)
# TODO: can the implementation of optimize_acqf_list be helpful to simultaneously evaluate from a list of acquisition functions

# TODO: keep in mind to change the gp_model from the handed over base_model class!
# TODO: is minimization effectively handled?
# TODO: implementation of q? can this be helpful to tune?



class GPOptimizer():

    def __init__(self, base_model: 'BaseGPModel') -> None:
        self.base_model = base_model
        self.acq_func = AcquisitionFunctionFactory.create_acquisition_function(acq_function_type='LogExp_Improvement', gp_model=base_model.gp_model, train_Y= base_model.train_Y, maximization=True)
    
    def optimization_loop(self, num_restarts: int, raw_samples: int, convergence_criteria: str = 'obj_func_pred_opt', manual_input: bool = True, max_iterations: int = 100):
        #convergence of objective function AND prediction output OR of the acq_func_value?
        
        for iteration in range(max_iterations):
            print(iteration)
            candidate, acq_value = self.optimization_iteration(num_restarts=num_restarts, raw_samples=raw_samples)
        
            if manual_input:
                print(f"Next suggested x-point: {candidate.item():.4f}")
                while True:
                    try:
                        next_y_value = float(input("Enter observation for given x: "))

                        break 
                    except ValueError:
                        print("Invalid input. Please enter a valid floating-point number as an input")
            else:
                raise ValueError(f"Automatic Input not supported yet in the optimization loop!")
            

            next_y_value = self.convert_y_input_tensor(y_input=next_y_value)
            
            #check if retraining is necessary - then set flag accordingly - flag is handed over in dataset add function of the model!
            #where is the retraining added? This should be done after adding the next point - training function of the model can be used

            self.base_model.add_point_to_dataset(new_X = candidate, new_Y = next_y_value)

            self.base_model.train(num_epochs=100)

            # TODO: do i need the refine still, if the model is already trained before?
            # TODO: current guess: condition_ob_observation is not the correct way in handling the data here!
        

    def optimization_iteration(self, num_restarts: int, raw_samples: int):
        
        candidate, acq_value = optimize_acqf(
            acq_function=self.acq_func, 
            bounds=self.base_model.bounds_list,
            q=1,
            num_restarts=num_restarts, 
            raw_samples=raw_samples
        )

        return candidate, acq_value


    def convert_y_input_tensor(self, y_input: float):
        tensor_value = torch.tensor([[y_input]], dtype=torch.float64)
        return tensor_value

    def plot_acq_func(self):
        pass