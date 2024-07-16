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
from optimization.bayesian_optimizer import BayesianOptimizer

from botorch import fit_gpytorch_mll

print(50*"-")

laser_hardening = FunctionFactory

main_dataset = DataManager(dataset_func=laser_hardening.laser_heat_treatment)

main_dataset.load_initial_dataset(num_datapoints=5, bounds=[(20, 400), (200e-3 / 60, 3000e-3 / 60), (83e-6, 1000e-6)], minimization_flags=[False, False], sampling_method="grid", noise_level=0)

multisingletaskgp = MultiSingletaskGPInitializer(main_dataset)

multisingletaskgp.initialize_model()

multisingletaskgp.train_initially_gp_model()

bayesian_optimizer = BayesianOptimizer(multiobjective_model=multisingletaskgp)

bayesian_optimizer.optimization_loop(num_iterations=10)


# sin_x = FunctionFactory


# # TODO this needs to be adjusted here: 

# main_dataset = DataManager(sin_x.multi_inputs)

# main_dataset.load_initial_dataset(num_datapoints=5, bounds=[(0, 6), (0, 2), (0, 3), (0, 2), (2, 3)], minimization_flags=[False, True, True], sampling_method="grid", noise_level=0)

# main_dataset.load_historic_dataset('smart_doe_bayesian_optimization\dataset_creation\pickle_files\datasets.pkl')

# multisingletaskgp = MultiSingletaskGPInitializer(main_dataset)

# multisingletaskgp.initialize_model()