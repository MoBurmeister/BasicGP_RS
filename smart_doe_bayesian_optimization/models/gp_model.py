from training import training
from models.gp_model_factory import GPModelFactory
from gpytorch.kernels import Kernel
from models.mll_factory import MLLFactory
from models.optimizer_factory import OptimizerFactory
from gpytorch.likelihoods import Likelihood
from visualization.visualization import GPVisualizer
import torch
from typing import Dict, Optional, List, Tuple

# TODO: are the Kernel and Likelihood variable types okay like that? should this be instead something else? -> Union
# TODO: checking function, if everything is setup correctly
# TODO: for the optimization iterations: it should be considered, that the train_X and train_Y data only should be updated, when the model is also updated for it. 
#       -> avoid the case, that the state of the data (X and Y) is different from the model state -> will result in problems for the visualization script

'''
Shape of the bounds list: 
torch.Size([2, d]) - ([min_x1, min_x2, ...],[max_x1, max_x2, ...])

tensor([[  0,  10, 100],[  6,  20, 200]]) for [(0, 6), (10, 20), (100, 200)]

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
            input_params = self.scaling_dict['inputs']
            method = input_params.get('method', 'default')
            if method == 'normalize':
                reversed_train_X = (reversed_train_X * (input_params['max'] - input_params['min'])) + input_params['min']
            elif method == 'standardize':
                reversed_train_X = (reversed_train_X * input_params['std']) + input_params['mean']
            elif method == 'default':
                # Data is unchanged, do nothing
                pass

        # Reverse scaling for train_Y if applicable
        if 'outputs' in self.scaling_dict:
            output_params = self.scaling_dict['outputs']
            method = output_params.get('method', 'default')
            if method == 'normalize':
                reversed_train_Y = (reversed_train_Y * (output_params['max'] - output_params['min'])) + output_params['min']
            elif method == 'standardize':
                reversed_train_Y = (reversed_train_Y * output_params['std']) + output_params['mean']
            elif method == 'default':
                # Data is unchanged, do nothing
                pass

        return reversed_train_X, reversed_train_Y
        

    def train(self,num_epochs):
        # TODO: Unnecessary reassignment of the gp_model -> since it is already modified in the training function (self.gp_model is handed over)
        self.gp_model = training.training_gp_model(gp_model=self.gp_model, optimizer=self.optimizer, mll=self.mll, train_X=self.train_X, train_Y=self.train_Y, num_epochs=num_epochs)

    def visualize_trained_model(self, rescale_vis: bool = False):
        #fig_dict_scaled, fig_dict_unscaled = 
        return GPVisualizer.visualize_model_pdp_with_uncertainty(self.gp_model, self.train_X, self.train_Y, scaling_dict=self.scaling_dict, rescale_vis=rescale_vis)
    
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
        return torch.tensor([lower_bounds, upper_bounds]) 

    #def optimizer(self, )