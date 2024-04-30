from training import training
from models.gp_model_factory import GPModelFactory
from gpytorch.kernels import Kernel
from models.mll_factory import MLLFactory
from models.optimizer_factory import OptimizerFactory
from gpytorch.likelihoods import Likelihood
import torch
from typing import Dict, Optional

# TODO: are the Kernel and Likelihood variable types okay like that? should this be instead something else? -> Union
# TODO: checking function, if everything is setup correctly

class BaseGPModel():
    def __init__(self, model_type: str, mll_type: str, optimizer_type: str, kernel: Kernel, train_X: torch.Tensor, train_Y: torch.Tensor, likelihood: Likelihood, scaling_dict: Optional[Dict] = None , optimizer_kwargs: dict=None):
        self.train_X = train_X
        self.train_Y = train_Y
        self.kernel = kernel
        self.likelihood = likelihood
        self.scaling_dict = scaling_dict
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

        # Reverse scaling for inputs
        scaling_info_inputs = self.scaling_dict['inputs']
        mean_inputs = scaling_info_inputs['mean']
        std_inputs = scaling_info_inputs['std']
        reversed_train_X = (self.train_X * std_inputs) + mean_inputs

        # Reverse scaling for outputs
        scaling_info_outputs = self.scaling_dict['outputs']
        mean_outputs = scaling_info_outputs['mean']
        std_outputs = scaling_info_outputs['std']
        reversed_train_Y = (self.train_Y * std_outputs) + mean_outputs

        return reversed_train_X, reversed_train_Y
        

    def train(self,num_epochs):
        # TODO: Unnecessary reassignment of the gp_model -> since it is already modified in the training function (self.gp_model is handed over)
        self.gp_model = training.training_gp_model(gp_model=self.gp_model, optimizer=self.optimizer, mll=self.mll, train_X=self.train_X, train_Y=self.train_Y, num_epochs=num_epochs)

    def visualize_trained_model(self):
        #fig_dic = 
        pass
        
    
    #def optimizer(self, )