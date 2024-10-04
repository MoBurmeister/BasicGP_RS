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
from smart_doe_bayesian_optimization.optimization.bayesian_optimizer import GPOptimizer
import matplotlib.pyplot as plt
from gpytorch.priors.torch_priors import GammaPrior
from utils.conversion_utils import matplotlib_to_png
from utils.config_parser_utils import config_parser

def setup_first_model():

    # xsinx = FunctionFactory

    # dataset_sum_sines = DatasetManager(dtype=torch.float64)

    # dataset_sum_sines.func_create_dataset(xsinx.sum_of_sines, num_datapoints=5, sampling_method="grid", noise_level=0, x1_range=(0,6), x2_range=(5,10), x3_range=(100,200))

    xsinx = FunctionFactory

    dataset_sum_sines = DatasetManager(dtype=torch.float64)
    dataset_sum_sines.func_create_dataset(xsinx.function_xsinx, num_datapoints=2, sampling_method="grid", noise_level=0, x1_range=(0,6))

    dict = config_parser("BasicGP_RS\smart_doe_bayesian_optimization\config_files\config_simple_GP.json")

    kernel = KernelFactory.create_kernel(dict["kernel_config"])

    likelihood = LikelihoodFactory.create_likelihood(dict["likelihood_config"])

    first_gp = BaseGPModel(dict["gp_model_config"], kernel, dataset_sum_sines.unscaled_data[0], dataset_sum_sines.unscaled_data[1], likelihood, bounds_list=dataset_sum_sines.bounds_list, scaling_dict=dict["scaling_config"])

    first_gp.train(num_epochs=100)

    first_gp.visualize_trained_model()

    matplotlib_to_png(first_gp.model_plots_dict['dim_1']['plot'], 'trained_model_fig.png', 'BasicGP_RS\\smart_doe_bayesian_optimization\\flask_app\\static\\images')

<<<<<<< HEAD
=======
    matplotlib_to_png(model_fig, 'trained_model_fig.png', 'BasicGP_RS\\smart_doe_bayesian_optimization\\flask_app\\static\\images')
    
>>>>>>> main
    return first_gp

def setup_optimizer(first_gp: BaseGPModel):

    gp_optimizer = GPOptimizer(base_model=first_gp, acq_func_type="LogExp_Improvement", is_maximization=True)

    gp_optimizer.plot_acq_func(num_points=100)

    matplotlib_to_png(gp_optimizer.acq_func_plot_dict["Dimension 1"], 'acq_func_fig.png', 'BasicGP_RS\\smart_doe_bayesian_optimization\\flask_app\\static\\images')

<<<<<<< HEAD
    return gp_optimizer
=======
    matplotlib_to_png(acq_fig, 'acq_function_fig.png', "BasicGP_RS\\smart_doe_bayesian_optimization\\flask_app\\static\\images")
>>>>>>> main

def perform_optimization_iteration(optimizer: GPOptimizer, observation: float):

    optimizer.optimization_iteration(observation=observation, num_restarts=40, raw_samples=400)

    optimizer.base_model.visualize_trained_model()

<<<<<<< HEAD
    matplotlib_to_png(optimizer.base_model.model_plots_dict['dim_1']['plot'], 'trained_model_fig.png', 'BasicGP_RS\\smart_doe_bayesian_optimization\\flask_app\\static\\images')

    optimizer.plot_acq_func(num_points=100)

    matplotlib_to_png(optimizer.acq_func_plot_dict["Dimension 1"], 'acq_func_fig.png', 'BasicGP_RS\\smart_doe_bayesian_optimization\\flask_app\\static\\images')
=======
    model_fig = optimizer.base_model.model_plots_dict['dim_1']['plot']    

    matplotlib_to_png(model_fig, 'trained_model_fig.png', 'BasicGP_RS\\smart_doe_bayesian_optimization\\flask_app\\static\\images')

    optimizer.plot_acq_func(num_points=100)

    acq_fig = optimizer.acq_func_plot_dict["Dimension 1"]

    matplotlib_to_png(acq_fig, 'acq_function_fig.png', 'BasicGP_RS\\smart_doe_bayesian_optimization\\flask_app\\static\\images')

    next_value, acq_value = optimizer.optimization_iteration(num_restarts=num_restarts, raw_samples=raw_samples)

    return next_value
>>>>>>> main
