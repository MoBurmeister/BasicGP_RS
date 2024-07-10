import models
import torch
from models.kernel_factory import KernelFactory
from gpytorch.priors import NormalPrior
from models.likelihood_factory import LikelihoodFactory
from data.function_factory import FunctionFactory
from gpytorch.constraints import GreaterThan
from models.optimizer_factory import OptimizerFactory
import matplotlib.pyplot as plt
from gpytorch.priors.torch_priors import GammaPrior
from utils.config_parser_utils import config_parser
from data.create_dataset import DataManager
from utils.checking_utils import check_type
from models.model_initializer.multi_singletaskgp_initializer import MultiSingletaskGPInitializer  

from botorch import fit_gpytorch_mll

print(50*"-")

laser_hardening = FunctionFactory

main_dataset = DataManager()

main_dataset.load_initial_dataset(laser_hardening.laser_heat_treatment, num_datapoints=5, bounds=[(20, 400), (200e-3 / 60, 3000e-3 / 60), (83e-6, 1000e-6)], sampling_method="grid", noise_level=0)

multisingletaskgp = MultiSingletaskGPInitializer(main_dataset)

multisingletaskgp.initialize_model()

multisingletaskgp.train_gp_model()






# sin_x = FunctionFactory

# main_dataset = DataManager()

# main_dataset.load_initial_dataset(sin_x.multi_inputs, num_datapoints=5, bounds=[(0, 6), (0, 2), (0, 3), (0, 2), (2, 3)], sampling_method="grid", noise_level=0)

# main_dataset.load_historic_dataset('smart_doe_bayesian_optimization\dataset_creation\pickle_files\datasets.pkl')

# multisingletaskgp = MultiSingletaskGPInitializer(main_dataset)

# multisingletaskgp.initialize_model()




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

# xsinx = FunctionFactory

# dataset_sum_sines = DatasetManager(dtype=torch.float64)

# dataset_sum_sines.func_create_dataset(xsinx.sum_of_sines, num_datapoints=5, sampling_method="grid", noise_level=0, x1_range=(0,6), x2_range=(5,10), x3_range=(100,200))

# dict = config_parser("BasicGP_RS\smart_doe_bayesian_optimization\config_files\config_simple_GP.json")

# kernel = KernelFactory.create_kernel(dict["kernel_config"])

# likelihood = LikelihoodFactory.create_likelihood(dict["likelihood_config"])

# first_gp = BaseGPModel(dict["gp_model_config"], kernel, dataset_sum_sines.unscaled_data[0], dataset_sum_sines.unscaled_data[1], likelihood, bounds_list=dataset_sum_sines.bounds_list, scaling_dict=dict["scaling_config"])

# first_gp.train(num_epochs=100)  

# first_gp.visualize_trained_model()

# optimizer = GPOptimizer(base_model=first_gp, acq_func_type="LogExp_Improvement", is_maximization=True)

# print(optimizer.next_proposed_parameter_setting)

# optimizer.optimization_iteration(observation=1.81)

# print(optimizer.next_proposed_parameter_setting)

# optimizer.optimization_iteration(observation=2.1)

# print("exit")
