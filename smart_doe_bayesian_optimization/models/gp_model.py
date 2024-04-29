from training import training
from models.gp_model_factory import GPModelFactory
from gpytorch.kernels import Kernel
from models.mll_factory import MLLFactory
from models.optimizer_factory import OptimizerFactory
from gpytorch.likelihoods import Likelihood
import torch

# TODO: are the Kernel and Likelihood variable types okay like that? should this be instead something else? -> Union
# TODO: checking function, if everything is setup correctly

class BaseGPModel():
    def __init__(self, model_type: str, mll_type: str, optimizer_type: str, kernel: Kernel, train_X: torch.Tensor, train_Y: torch.Tensor, likelihood: Likelihood, optimizer_kwargs: dict=None):
        self.train_X = train_X
        self.train_Y = train_Y
        self.kernel = kernel
        self.likelihood = likelihood
        self.gp_model = GPModelFactory.create_model(model_type, train_X, train_Y, kernel, likelihood)
        self.mll = MLLFactory.create_mll(type=mll_type, model=self.gp_model, likelihood=self.likelihood)
        self.optimizer = OptimizerFactory.create_optimizer(type=optimizer_type, model_parameters=self.gp_model.parameters(), **(optimizer_kwargs or {}))

    def to(self, device, dtype):
        self.gp_model = self.gp_model.to(device=device, dtype=dtype)
        self.mll = self.mll.to(device=device, dtype=dtype)

    def train(self,num_epochs):
        # TODO: Unnecessary reassignment of the gp_model -> since it is already modified in the training function (self.gp_model is handed over)
        self.gp_model = training.training_gp_model(gp_model=self.gp_model, optimizer=self.optimizer, mll=self.mll, train_X=self.train_X, train_Y=self.train_Y, num_epochs=num_epochs)

    def visualize_trained_model(self, )
        
    
    #def optimizer(self, )