# from training import training
# from models.gp_model_factory import GPModelFactory
# from gpytorch.kernels import Kernel
# from models.mll_factory import MLLFactory
# from models.optimizer_factory import OptimizerFactory
# from gpytorch.likelihoods import Likelihood
# from visualization.visualization import GPVisualizer
# from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
# from botorch.fit import fit_gpytorch_mll
# from botorch.models.transforms.outcome import OutcomeTransform, Standardize
# from botorch.models.transforms.input import InputTransform, Normalize
# from utils.checking_utils import check_same_dimension
# import torch
# from typing import Dict, Optional, List, Tuple

from abc import ABC, abstractmethod

from botorch.acquisition.objective import PosteriorTransform
from botorch.posteriors import Posterior
from torch import Tensor
from data.create_dataset import DataManager
from data.dataset import Dataset
from typing import Any, List, Tuple, Union
from botorch.models.model import Model

# TODO: are the Kernel and Likelihood variable types okay like that? should this be instead something else? -> Union
# TODO: checking function, if everything is setup correctly
# TODO: for the optimization iterations: it should be considered,    the train_X and train_Y data only should be updated, when the model is also updated for it. 
#       -> avoid the case, that the state of the data (X and Y) is different from the model state -> will result in problems for the visualization script

'''
Shape of the bounds list: 
torch.Size([2, d]) - ([min_x1, min_x2, ...],[max_x1, max_x2, ...])

tensor([[  0,  10, 100],[  6,  20, 200]]) for original input [(0, 6), (10, 20), (100, 200)]
'''

class BaseModel(ABC):
    def __init__(self, dataset_manager: DataManager):
        self.dataset_manager = dataset_manager
       
    @abstractmethod
    def train_gp_model(self):
        pass
    

    

    def print_model_info(self):
        #print out the model information via print statement in the model class
        pass

        




# class BaseGPModel():
#     def __init__(self, model_setup_dict: dict, kernel: Kernel, train_X: torch.Tensor, train_Y: torch.Tensor, likelihood: Likelihood, bounds_list: List[Tuple], scaling_dict: dict = None):
#         self.train_X = train_X
#         self.train_Y = train_Y
#         self.kernel = kernel
#         self.likelihood = likelihood
#         self.bounds_list = self.create_bounds_tensor(bounds_list=bounds_list)
#         self.gp_model_type = model_setup_dict.get('model_type')
#         self.scaling_dict = scaling_dict
#         self.outcome_transform, self.input_transform = None, None
#         self.gp_model = None
#         self.mll_type = model_setup_dict.get('mll_type')
#         self.mll = None
#         #the optimizer here is only necessary for the traditional torch training without full convergence
#         self.optimizer = None
#         self.model_plots_dict = {}

#         self.print_model_info()

#         # TODO: What about the "to" function? when applicable? Relevant for later use in gpu station settings!

#     def to(self, device, dtype):
#         self.gp_model = self.gp_model.to(device=device, dtype=dtype)
#         self.mll = self.mll.to(device=device, dtype=dtype)

#     def setup_transformations(self, scaling_dict: dict):

#         input_scaling_method = scaling_dict.get('input')
#         output_scaling_method = scaling_dict.get('output')

#         input_transformation_method = None
#         outcome_transformation_method = None

#         if input_scaling_method == 'normalize':
#             input_transformation_method = Normalize(d=self.train_X.shape[1])
#         elif input_scaling_method is None:
#             pass
#         else:
#             raise ValueError("Invalid input scaling method. Expected 'normalize' or None.")

#         if output_scaling_method == 'standardize':
#             outcome_transformation_method = Standardize(m=self.train_Y.shape[1])
#         elif output_scaling_method is None:
#             pass
#         else:
#             raise ValueError("Invalid output scaling method. Expected 'standardize' or None.")

#         return outcome_transformation_method, input_transformation_method   

# # TODO: option to not completely retrain and re-initiate the model? dict can be handed over? How long can the training time even be?

#     def train(self, num_epochs: int, convergence_training: bool = True):
#         '''
#         Function for the training of the model with the data stored in the current train_X, train_Y
#         '''

#         # TODO: printout for training with train settings such as transformations (input/output), convergence criteria, bounds, ...

#         # TODO: clarify training! useable methods: training with traditional pytorch methods (e.g. using adam as an optimizer) or using the L-BFGS-B optimizer via fit_gpytorch_mll or with condition_model?
#         if convergence_training:
#             print(f"Performing botorch convergence training with L-BFGS-B optimizer via fit_gpytorch_mll!")
#             # TODO: IMPORTANT: The scaling metrics are stored in the transformations! So e.g. after inital setup they are empty and with the first model initialization they are filled! 
#             #       So if they change drastically with a new datapoint, they need to be re-initiated and therefore re-calculated!
#             # TODO: If with a new datapoint the distribution of the data does not change significantly, then the transformations do not necessarily need to be re-initiated. This can be made optional, e.g. based on a schedule
#             self.outcome_transform, self.input_transform = self.setup_transformations(scaling_dict=self.scaling_dict)
#             print(f"Training with input transformation: {self.input_transform} and outcome transformation: {self.outcome_transform}.")
#             self.gp_model = GPModelFactory.create_model(model_type = self.gp_model_type, train_X=self.train_X, train_Y=self.train_Y, kernel=self.kernel, likelihood=self.likelihood, outcome_transform=self.outcome_transform, input_transform=self.input_transform)
#             self.mll = MLLFactory.create_mll(type=self.mll_type, model=self.gp_model, likelihood=self.likelihood)
#             fit_gpytorch_mll(self.mll)
#             # TODO: printout for training progress and also later training results?

#         else:
#             raise ValueError("Traditional training is not yet implemented! Please set convergence_training to True!")
#             #print(f"Performing traditional training with {self.optimizer} as an optimizer, training over {num_epochs} epochs!")
#             # TODO: fit_gpytorch_mll_torch can also be a useable function here - what does it implement/realise better or worse? what should be used here?
#             # TODO: optimizer needs to be re-initiated here!
#             #self.gp_model = training.training_gp_model(gp_model=self.gp_model, optimizer=self.optimizer, mll=self.mll, train_X=self.train_X, train_Y=self.train_Y, num_epochs=num_epochs)

#     def visualize_trained_model(self):
#         self.model_plots_dict = GPVisualizer.visualize_model_pdp_with_uncertainty(self.gp_model, self.train_X, self.train_Y, bounds_list=self.bounds_list, num_points=100)
        

#     # TODO: Implementation of adding multiple points to the dataset - or already possible?

#     def add_point_to_dataset(self, new_X: torch.Tensor, new_Y: torch.Tensor):

#         check_same_dimension(tensor1=new_X, tensor2=self.train_X)

#         self.train_X = torch.cat([self.train_X, new_X], 0)
#         self.train_Y = torch.cat([self.train_Y, new_Y], 0)

        
#     def create_bounds_tensor(self, bounds_list):
#         """
#         Convert a list of bounds into a 2 x d tensor format for Gaussian Process models.

#         Parameters:
#         - bounds_list (list of tuples): Each tuple contains (lower_bound, upper_bound) for a dimension.

#         Returns:
#         - torch.Tensor: A 2 x d tensor where the first row contains lower bounds and the second row contains upper bounds.
#         """
#         # Extract lower and upper bounds
#         lower_bounds = [lb for lb, ub in bounds_list]
#         upper_bounds = [ub for lb, ub in bounds_list]

#         # Create and return the tensor
#         return torch.tensor([lower_bounds, upper_bounds], dtype=torch.float64) 

#     def get_last_added_datapoint(self, return_train_X: bool = True):
#         """
#         Function to return the last added datapoint from the training dataset, based on the assumption that the last added datapoint is the last element in the training dataset.

#         Parameters:
#         - return_X (bool): If True, return the last added datapoint from train_X. If False, return the last added datapoint from train_Y.

#         Returns:
#         - torch.Tensor: The last added datapoint.
#         """
#         if return_train_X:
#             return self.train_X[-1]
#         else:
#             return self.train_Y[-1]
        
#     def print_model_info(self):
#         """
#         Prints detailed information about the Gaussian Process model configuration.
#         """
#         print("========== Gaussian Process Model Setup ==========")
#         print(f"Model Type: {self.gp_model_type}")
#         print(f"Kernel Type: {type(self.kernel).__name__}")
#         # TODO: printout for the kernel parameters
#         print(f"Likelihood Type: {type(self.likelihood).__name__}")
#         # TODO: printout for the likelihood parameters
#         print(f"Marginal Log Likelihood Type: {self.mll_type}")

#         print(f"Training Data Points (train_X): {self.train_X.shape[0]}")
#         print(f"Input Dimensions (train_X): {self.train_X.shape[1]}")
#         print(f"Output Dimensions (train_Y): {self.train_Y.shape[1]}")
        
#         if self.scaling_dict.get('input'):
#             print(f"Input Transformation: {self.scaling_dict.get('input')}")
#         else:
#             print("No input transformation applied.")

#         if self.scaling_dict.get('output'):
#             print(f"Outcome Transformation: {self.scaling_dict.get('output')}")
#         else:
#             print("No outcome transformation applied.")

#         # print("Bounds for input features:")
#         # for i, bounds in enumerate(self.bounds_list):
#         #     print(f"  Dimension {i + 1}: Lower = {bounds[0]}, Upper = {bounds[1]}")

#         print("=======================================================")
