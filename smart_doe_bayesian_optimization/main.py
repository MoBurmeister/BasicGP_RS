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
import matplotlib.pyplot as plt

xsinx = FunctionFactory

dataset_xsinx = DatasetManager(dtype=torch.float64)
dataset_xsinx.func_create_dataset(xsinx.function_xsinx, num_datapoints=5, sampling_method="grid", noise_level=0.1, scaling_input='normalize', scaling_output='standardize', x1_range=(0,6))

#Create a prior for the lengthscale
lengthscale_prior = NormalPrior(loc=1.0, scale=0.1)

# TODO: How to check for the ard_num_dims? Where to setup and ensure correct running?

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

first_gp = BaseGPModel("SingleTaskGP", "ExactMarginalLogLikelihood", "adam", rbf_kernel, dataset_xsinx.scaled_data[0], dataset_xsinx.scaled_data[1], gp_likelihood, bounds_list=dataset_xsinx.bounds_list, scaling_dict=dataset_xsinx.scaling_dict)

first_gp.train(num_epochs=1500)

plots = first_gp.visualize_trained_model(rescale_vis=True)

# fig = plots[0]

# print(plots)

# plt.figure(figsize=(8, 6))  # Optionally set the figure size
# plt.show()


