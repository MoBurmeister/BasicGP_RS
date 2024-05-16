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
from optimization.optimization import GPOptimizer
import matplotlib.pyplot as plt
from gpytorch.priors.torch_priors import GammaPrior
from utils.config_parser_utils import config_parser

from botorch import fit_gpytorch_mll

# xsinx = FunctionFactory

# dataset_xsinx = DatasetManager(dtype=torch.float64)
# dataset_xsinx.func_create_dataset(xsinx.function_xsinx, num_datapoints=2, sampling_method="grid", noise_level=0, x1_range=(0,6))

# #Create a prior for the lengthscale
# lengthscale_prior = NormalPrior(loc=1.0, scale=0.1)

# # TODO: How to check for the ard_num_dims? Where to setup and ensure correct running?

# # Instantiate an RBF Kernel with a lengthscale prior and ARD for 3 dimensions
# rbf_kernel = KernelFactory.create_kernel(
#     'Matern', 
#     nu=2.5, 
#     lengthscale_prior=GammaPrior(3.0, 6.0)
# )

# gp_likelihood = LikelihoodFactory.create_likelihood(
#     'Gaussian',
#     noise_constraint = GreaterThan(1e-5)
# )

# scaling_dict = {
#     'input': 'normalize',
#     'output': 'standardize'
# }

# first_gp = BaseGPModel("SingleTaskGP", "ExactMarginalLogLikelihood", "adam", rbf_kernel, dataset_xsinx.unscaled_data[0], dataset_xsinx.unscaled_data[1], gp_likelihood, bounds_list=dataset_xsinx.bounds_list, scaling_dict=scaling_dict, optimizer_kwargs={"lr":0.1})

# first_gp.train(num_epochs=100)

# first_gp.visualize_trained_model()

# gp_optimizer = GPOptimizer(base_model=first_gp, acq_func_type="LogExp_Improvement", is_maximization=True)

# gp_optimizer.optimization_loop(num_restarts=40, raw_samples=400, max_iterations=1)

# first_gp.visualize_trained_model()

# gp_optimizer.plot_acq_func(num_points=100)

# plot = gp_optimizer.acq_func_plot_dict.get(0)

# plt.show()

xsinx = FunctionFactory

dataset_xsinx = DatasetManager(dtype=torch.float64)
dataset_xsinx.func_create_dataset(xsinx.function_xsinx, num_datapoints=2, sampling_method="grid", noise_level=0, x1_range=(0,6))

dict = config_parser("BasicGP_RS\smart_doe_bayesian_optimization\config_files\config_simple_GP.json")

kernel = KernelFactory.create_kernel(dict["kernel_config"])

likelihood = LikelihoodFactory.create_likelihood(dict["likelihood_config"])

first_gp = BaseGPModel(dict["gp_model_config"], kernel, dataset_xsinx.unscaled_data[0], dataset_xsinx.unscaled_data[1], likelihood, bounds_list=dataset_xsinx.bounds_list, scaling_dict=dict["scaling_config"])
