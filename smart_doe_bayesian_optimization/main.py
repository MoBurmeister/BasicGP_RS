import models
import torch
from models.gp_model import BaseGPModel
from models.kernel_factory import KernelFactory
from gpytorch.priors import NormalPrior
from models.likelihood_factory import LikelihoodFactory
from data.create_dataset import DatasetManager
from data.function_factory import FunctionFactory
from gpytorch.constraints import GreaterThan
from models.optimizer_factory import OptimizerFactory


#create first sample dataset with x*sin(x)
#first setup function x*sin(x) from function factory

xsinx = FunctionFactory

dataset_xsinx = DatasetManager(dtype=torch.float64)
dataset_xsinx.func_create_dataset(xsinx.sum_of_sines, num_datapoints=5, sampling_method="random", noise_level=0.1, x1_range=(0,6), x2_range=(10,20), x3_range=(100,200))

print(f"initial x, unchanged: {dataset_xsinx.unscaled_data[0]}")

#Create a prior for the lengthscale
lengthscale_prior = NormalPrior(loc=1.0, scale=0.1)

# Instantiate an RBF Kernel with a lengthscale prior and ARD for 3 dimensions
rbf_kernel = KernelFactory.create_kernel(
    'RBF', 
    ard_num_dims=1, 
    lengthscale_prior=lengthscale_prior, 
)

gp_likelihood = LikelihoodFactory.create_likelihood(
    'Gaussian',
    noise_constraint = GreaterThan(1e-4)
)

first_gp = BaseGPModel("SingleTaskGP", "ExactMarginalLogLikelihood", "adam", rbf_kernel, dataset_xsinx.scaled_data[0], dataset_xsinx.scaled_data[1], gp_likelihood, scaling_dict=dataset_xsinx.scaling_dict)

rescaled_X, rescaled_Y = first_gp.reverse_scale()

print(f"scaled data: {rescaled_X}")