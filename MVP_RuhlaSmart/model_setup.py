#necessary imports
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.utils import standardize
from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from gpytorch.priors.torch_priors import GammaPrior

#setup ground truth function:

#define the original functions:
def gt_function(x_values: torch.tensor):
    #original function: x*sin(x) with added noise
    # The main function as per your image
    main_term = -torch.cos(x_values * torch.pi / 2.5) * torch.sin(x_values) + 1
    # Adding Gaussian noise
    noise = 0.05 * torch.randn_like(x_values)
    return main_term + noise

#script to setup the model with all required additions
def setup_datasets(dtype, device, gt_function:callable, num_datapoints: int, lower_bound: float, upper_bound: float):
    #define the bounds
    bounds = torch.tensor([[lower_bound], [upper_bound]], dtype=torch.float32)
    #use regular spaced points on the interval [0, 6] -> here 4 equally spaced points between 0 and 6
    x_dataset = torch.linspace(lower_bound, upper_bound, num_datapoints, dtype=dtype, device=device)
    #training data needs to be explicitly multi-dimensional, here from 1-dimensional to 2-dimensional
    x_dataset = x_dataset.unsqueeze(1)
    #calculate the y-values
    y_dataset = gt_function(x_dataset)
    #return the datasets and the bounds
    return x_dataset, y_dataset, bounds

def setup_model(train_X: torch.tensor, train_Y: torch.tensor):
    #setup the likelihood:
    likelihood = GaussianLikelihood(noise_constraint=GreaterThan(0.001))
    #setup the kernel: 
    kernel = ScaleKernel(MaternKernel(nu=2.5, lengthscale_prior=GammaPrior(3.0, 6.0)),outputscale_prior=GammaPrior(2.0, 0.15))    
    #model setup
    gp_model = SingleTaskGP(covar_module=kernel, likelihood=likelihood, train_X=train_X, train_Y=train_Y)
    #setup the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood=likelihood, model=gp_model)
    # Set to the same dtype and device as train_X
    gp_model = gp_model.to(train_X.device, dtype=train_X.dtype)
    mll = mll.to(train_X.device, dtype=train_X.dtype)
    #define optimizer
    optimizer = Adam([{"params": gp_model.parameters()}], lr=0.1)

    return gp_model, optimizer, mll

