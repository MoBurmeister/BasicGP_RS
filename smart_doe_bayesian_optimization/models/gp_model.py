from training import training
from models.gp_model_factory import GPModelFactory
from gpytorch.kernels import Kernel
from models.mll_factory import MLLFactory
from models.optimizer_factory import OptimizerFactory
from gpytorch.likelihoods import Likelihood
from visualization.visualization import GPVisualizer
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
import torch
from typing import Dict, Optional, List, Tuple

# TODO: are the Kernel and Likelihood variable types okay like that? should this be instead something else? -> Union
# TODO: checking function, if everything is setup correctly
# TODO: for the optimization iterations: it should be considered, that the train_X and train_Y data only should be updated, when the model is also updated for it. 
#       -> avoid the case, that the state of the data (X and Y) is different from the model state -> will result in problems for the visualization script
# TODO: "default" not really implemented?

'''
Shape of the bounds list: 
torch.Size([2, d]) - ([min_x1, min_x2, ...],[max_x1, max_x2, ...])

tensor([[  0,  10, 100],[  6,  20, 200]]) for [(0, 6), (10, 20), (100, 200)]

Typical scaling dict:
{'inputs': {'method': 'normalize', 
            'params': {'min': tensor([0]), 
            'max': tensor([6])}, 
            'scaled_bounds': tensor([[0.], [1.]]), 
            'original_bounds': tensor([[0], [6]])}, 
'outputs': {'method': 'standardize', 
            'params': {'mean': tensor([-0.4406], dtype=torch.float64), 
            'std': tensor([1.0026], dtype=torch.float64)}, 
            'scaled_bounds': None, 
            'original_bounds': None}}
'''

class BaseGPModel():
    def __init__(self, model_type: str, mll_type: str, optimizer_type: str, kernel: Kernel, train_X: torch.Tensor, train_Y: torch.Tensor, likelihood: Likelihood, bounds_list: List[Tuple], scaling_dict: Optional[Dict] = None , optimizer_kwargs: dict=None):
        self.train_X = train_X
        self.train_Y = train_Y
        self.kernel = kernel
        self.likelihood = likelihood
        self.scaling_dict = scaling_dict
        self.bounds_list = self.create_bounds_tensor(bounds_list=bounds_list)
        self.gp_model = GPModelFactory.create_model(model_type, train_X, train_Y, kernel, likelihood)
        self.mll = MLLFactory.create_mll(type=mll_type, model=self.gp_model, likelihood=self.likelihood)
        self.optimizer = OptimizerFactory.create_optimizer(type=optimizer_type, model_parameters=self.gp_model.parameters(), **(optimizer_kwargs or {}))

    def to(self, device, dtype):
        self.gp_model = self.gp_model.to(device=device, dtype=dtype)
        self.mll = self.mll.to(device=device, dtype=dtype)

    def reverse_scale(self):
        """
        Reverse scales the train_X and train_Y attributes based on the scaling information stored in scaling_dict.
        Returns the reverse scaled train_X and train_Y tensors without modifying the original class attributes.
        """
        if not self.scaling_dict:
            print(f"ATTENTION: The Scaling dic is empty, just returning the inital data")
            return self.train_X.clone(), self.train_Y.clone()
    
        # Initialize reverse scaled data with current scaled data
        reversed_train_X = self.train_X.clone()
        reversed_train_Y = self.train_Y.clone()

        # Reverse scaling for train_X if applicable
        if 'inputs' in self.scaling_dict:
            input_params = self.scaling_dict['inputs']['params']
            method = self.scaling_dict['inputs']['method']
            if method == 'normalize':
                reversed_train_X = (reversed_train_X * (input_params['max'] - input_params['min'])) + input_params['min']
            elif method == 'standardize':
                reversed_train_X = (reversed_train_X * input_params['std']) + input_params['mean']
            elif method == 'default':
                # Data is unchanged, do nothing
                pass

        # Reverse scaling for train_Y if applicable
        if 'outputs' in self.scaling_dict:
            output_params = self.scaling_dict['outputs']['params']
            method = self.scaling_dict['outputs']['method']
            if method == 'normalize':
                reversed_train_Y = (reversed_train_Y * (output_params['max'] - output_params['min'])) + output_params['min']
            elif method == 'standardize':
                reversed_train_Y = (reversed_train_Y * output_params['std']) + output_params['mean']
            elif method == 'default':
                # Data is unchanged, do nothing
                pass

        return reversed_train_X, reversed_train_Y
        

    def train(self, num_epochs: int, convergence_training: bool = True):
        '''
        Function for the training of the model with the data stored in the  
        '''
        # TODO: clarify training! useable methods: training with traditional pytorch methods (e.g. using adam as an optimizer) or using the L-BFGS-B optimizer via fit_gpytorch_mll or with condition_model?
        if convergence_training:
            print(f"Performing botorch convergence training with L-BFGS-B optimizer via fit_gpytorch_mll!")

        else:
            print(f"Performing traditional training with {self.optimizer} as an optimizer, training over {num_epochs} epochs!")
            # TODO: Unnecessary reassignment of the gp_model -> since it is already modified in the training function (self.gp_model is handed over)
            # TODO: fit_gpytorch_mll_torch can also be a useable function here - what does it implement/realise better or worse? what should be used here?
            self.gp_model = training.training_gp_model(gp_model=self.gp_model, optimizer=self.optimizer, mll=self.mll, train_X=self.train_X, train_Y=self.train_Y, num_epochs=num_epochs)

    def visualize_trained_model(self, rescale_vis: bool = False):
        #fig_dict_scaled, fig_dict_unscaled = 
        return GPVisualizer.visualize_model_pdp_with_uncertainty(self.gp_model, self.train_X, self.train_Y, scaling_dict=self.scaling_dict, rescale_vis=rescale_vis)
    

    # TODO: Implementation of adding multiple points to the dataset
    # TODO: Is this correct here/the good way, that only the new_Y is scaled? Because the new X already is acquired as scaled from the model as a suggestion

    def add_point_to_dataset(self, new_X: torch.Tensor, new_Y: torch.Tensor, rescale_all: bool):
        
        if rescale_all:

            reversed_train_X, reversed_train_Y = self.reverse_scale()

            reversed_train_X = torch.cat([reversed_train_X, new_X], 0)
            reversed_train_Y = torch.cat([reversed_train_Y, new_Y], 0)

            #perform the scaling: calculate all metrics new, perform scaling, adjust scaling dic and change the currently saved training data
            self.train_X = self.perform_scaling(reversed_train_X, 'inputs')
            self.train_Y = self.perform_scaling(reversed_train_Y, 'outputs')

        else:
            
            # TODO: not complete correct implementation here?!

            new_Y = self.scale_point(point=new_Y, data_type='outputs')
            self.train_X = torch.cat([self.train_X, new_X], 0)
            self.train_Y = torch.cat([self.train_Y, new_Y], 0)

    def scale_point(self, point: torch.Tensor, data_type: str):
        if data_type not in self.scaling_dict:
            raise ValueError(f"No scaling parameters found for data type '{data_type}'")
        
        scaling_info = self.scaling_dict[data_type]
        method = scaling_info['method']
        params = scaling_info['params']

        if method == 'normalize':
            min_val = params['min']
            max_val = params['max']
            scaled_point = (point - min_val) / (max_val - min_val)
        elif method == 'standardize':
            mean_val = params['mean']
            std_val = params['std']
            scaled_point = (point - mean_val) / std_val
        elif method == 'default':
            scaled_point = point
        else:
            raise ValueError(f"Unsupported scaling method '{method}'")
        
        return scaled_point
    
    def rescale_point(self, point: torch.Tensor, data_type: str):
        if data_type not in self.scaling_dict:
            raise ValueError(f"No scaling parameters found for data type '{data_type}'")

        scaling_info = self.scaling_dict[data_type]
        method = scaling_info['method']
        params = scaling_info['params']

        if method == 'normalize':
            min_val = params['min']
            max_val = params['max']
            rescaled_point = point * (max_val - min_val) + min_val
        elif method == 'standardize':
            mean_val = params['mean']
            std_val = params['std']
            rescaled_point = point * std_val + mean_val
        elif method == 'default':
            rescaled_point = point
        else:
            raise ValueError(f"Unsupported scaling method '{method}'")
        
        return rescaled_point

    def perform_scaling(self, data: torch.Tensor, data_type: str):

        scaling_info = self.scaling_dict[data_type]
        method = scaling_info['method']

        if method == 'normalize':
            min_val = scaling_info['params']['min']
            max_val = scaling_info['params']['max']
            range_val = max_val - min_val
            # Avoid division by zero if the range is 0
            range_val[range_val == 0] = 1
            scaled_data = (data - min_val) / range_val
        elif method == 'standardize':
            mean_val = data.mean(dim=0)
            std_val = data.std(dim=0)
            # Avoid division by zero if std deviation is 0
            std_val[std_val == 0] = 1
            scaled_data = (data - mean_val) / std_val

            # Update scaling dictionary
            if data_type == 'inputs':
                original_bounds = torch.tensor(self.bounds_list).t()
                scaled_lower_bounds = (original_bounds[0] - mean_val) / std_val
                scaled_upper_bounds = (original_bounds[1] - mean_val) / std_val
                scaled_bounds = torch.stack((scaled_lower_bounds, scaled_upper_bounds), dim=0)
            else:
                scaled_bounds = None

            # Updating the scaling dictionary
            self.scaling_dict[data_type] = {
                'method': 'standardize',
                'params': {'mean': mean_val, 'std': std_val},
                'scaled_bounds': scaled_bounds
            }

        elif method == 'default':
            scaled_data = data
        else:
            raise ValueError(f"Unsupported scaling method '{method}'")

        return scaled_data
        
    def create_bounds_tensor(self, bounds_list):
        """
        Convert a list of bounds into a 2 x d tensor format for Gaussian Process models.

        Parameters:
        - bounds_list (list of tuples): Each tuple contains (lower_bound, upper_bound) for a dimension.

        Returns:
        - torch.Tensor: A 2 x d tensor where the first row contains lower bounds and the second row contains upper bounds.
        """
        # Extract lower and upper bounds
        lower_bounds = [lb for lb, ub in bounds_list]
        upper_bounds = [ub for lb, ub in bounds_list]

        # Create and return the tensor
        return torch.tensor([lower_bounds, upper_bounds], dtype=torch.float64) 
