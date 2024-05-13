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
from utils.conversion_utils import matplotlib_to_png


from botorch import fit_gpytorch_mll

def setup_first_model():

    xsinx = FunctionFactory

    dataset_xsinx = DatasetManager(dtype=torch.float64)
    dataset_xsinx.func_create_dataset(xsinx.function_xsinx, num_datapoints=2, sampling_method="grid", noise_level=0, x1_range=(0,6))

    #Create a prior for the lengthscale
    lengthscale_prior = NormalPrior(loc=1.0, scale=0.1)

    # TODO: How to check for the ard_num_dims? Where to setup and ensure correct running?

    # Instantiate an RBF Kernel with a lengthscale prior and ARD for 3 dimensions
    rbf_kernel = KernelFactory.create_kernel(
        'Matern', 
        nu=2.5, 
        lengthscale_prior=GammaPrior(3.0, 6.0)
    )

    gp_likelihood = LikelihoodFactory.create_likelihood(
        'Gaussian',
        noise_constraint = GreaterThan(1e-5)
    )

    scaling_dict = {
        'input': 'normalize',
        'output': 'standardize'
    }

    first_gp = BaseGPModel("SingleTaskGP", "ExactMarginalLogLikelihood", "adam", rbf_kernel, dataset_xsinx.unscaled_data[0], dataset_xsinx.unscaled_data[1], gp_likelihood, bounds_list=dataset_xsinx.bounds_list, scaling_dict=scaling_dict, optimizer_kwargs={"lr":0.1})

    first_gp.train(num_epochs=100)

    first_gp.visualize_trained_model()

    model_fig = first_gp.model_plots_dict['dim_1']['plot']    

    matplotlib_to_png(model_fig, 'trained_model_fig.png', 'flask_app\static\images')
    
    return first_gp


def setup_optimization_loop(first_gp: BaseGPModel):

    gp_optimizer = GPOptimizer(base_model=first_gp, acq_func_type="LogExp_Improvement", is_maximization=True)

    next_value, _ = gp_optimizer.optimization_iteration(num_restarts=40, raw_samples=400)

    gp_optimizer.plot_acq_func(num_points=100)

    acq_fig = gp_optimizer.acq_func_plot_dict["Dimension 1"]

    matplotlib_to_png(acq_fig, 'acq_function_fig.png', 'flask_app\static\images')

    return gp_optimizer, next_value

def get_next_optimization_iteration(optimizer: GPOptimizer, input_value: float, original_x: torch.Tensor, num_restarts: int = 40, raw_samples: int = 400):

    input_value = optimizer.convert_y_input_tensor(input_value)

    print(input_value)
    print(type(input_value))
    print(original_x)
    print(type(original_x))

    optimizer.base_model.add_point_to_dataset(new_X = original_x, new_Y = input_value)

    optimizer.base_model.train(num_epochs=100)

    optimizer.base_model.visualize_trained_model()

    model_fig = optimizer.base_model.model_plots_dict['dim_1']['plot']    

    matplotlib_to_png(model_fig, 'trained_model_fig.png', 'flask_app\static\images')

    optimizer.plot_acq_func(num_points=100)

    acq_fig = optimizer.acq_func_plot_dict["Dimension 1"]

    matplotlib_to_png(acq_fig, 'acq_function_fig.png', 'flask_app\static\images')

    next_value, acq_value = optimizer.optimization_iteration(num_restarts=num_restarts, raw_samples=raw_samples)

    return next_value