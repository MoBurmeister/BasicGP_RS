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
from data.constraint_factory import WeldingConstraints
from botorch.test_functions.multi_objective import WeldedBeam
from botorch import fit_gpytorch_mll
from botorch.optim import gen_batch_initial_conditions

'''
Important bevore running an optimization:
- Check maximization flags
- Check the reference point (negate all values for minimization)
- Resulting Hypervolume is dependent on the outcome ranges! can be very different in size

'''

print(50*"-")

# vehicle_crash = FunctionFactory

# main_dataset = DataManager(dataset_func=vehicle_crash.generate_car_crash_synthetic_data)

# main_dataset.load_initial_dataset(num_datapoints=5, bounds=[(1.0, 3.0)] * 5, maximization_flags=[False, False, False], input_parameter_name=["x1", "x2", "x3", "x4", "x5"], output_parameter_name=["Mass", "A_inn", "Intrusion"])

# multisingletaskgp = MultiSingletaskGPInitializer(main_dataset)

# multisingletaskgp.initially_setup_model()    

# multisingletaskgp.train_initially_gp_model()

# #reference point handed over as negative values!
# bayesian_optimizer = BayesianOptimizer(multiobjective_model=multisingletaskgp, reference_point=torch.tensor([-1864.72022, -11.81993945, -0.2903999384], dtype=torch.float64))

# bayesian_optimizer.optimization_loop(num_max_iterations=50)

# print(bayesian_optimizer.multiobjective_model.dataset_manager.initial_dataset.input_data)
# print(bayesian_optimizer.multiobjective_model.dataset_manager.initial_dataset.output_data)

#########

# welding_beam = FunctionFactory

# main_dataset = DataManager(dataset_func=welding_beam.welding_beam)

# main_dataset.load_initial_dataset(num_datapoints=5, bounds = [(0.125, 5.0),(0.1, 10.0),(0.1, 10.0),(0.125, 5.0)], maximization_flags=[False, False], sampling_method="grid", noise_level=0)

# multisingletaskgp = MultiSingletaskGPInitializer(main_dataset)

# multisingletaskgp.initially_setup_model()    

# multisingletaskgp.train_initially_gp_model()

# bayesian_optimizer = BayesianOptimizer(multiobjective_model=multisingletaskgp, reference_point=torch.tensor([-40, -0.015], dtype=torch.float64))

# bayesian_optimizer.optimization_loop(num_iterations=100)


laser_hardening = FunctionFactory

main_dataset = DataManager(dataset_func=laser_hardening.laser_heat_treatment)

main_dataset.load_initial_dataset(num_datapoints=11, bounds=[(50, 150), (0.00333333, 0.00333333), (0.001, 0.001)], maximization_flags=[True, False], input_parameter_name=["laser_pwr", "laser_speed", "laser_width"], output_parameter_name=["hardening_time", "temp_div"], sampling_method="grid", noise_level=0)

#main_dataset.load_initial_dataset(num_datapoints=1500, bounds=[(20, 400), (200e-3 / 60, 3000e-3 / 60), (83e-6, 1000e-6)], maximization_flags=[True, False], input_parameter_name=["laser_pwr", "laser_speed", "laser_width"], output_parameter_name=["hardening_time", "temp_div"], sampling_method="grid", noise_level=0)

print(main_dataset.initial_dataset.input_data)

print(main_dataset.initial_dataset.output_data)

# multisingletaskgp = MultiSingletaskGPInitializer(main_dataset)

# multisingletaskgp.initially_setup_model()

# multisingletaskgp.train_initially_gp_model()

# bayesian_optimizer = BayesianOptimizer(multiobjective_model=multisingletaskgp)

# bayesian_optimizer.optimization_loop(num_max_iterations=1)

# def constraint_func(outputs: torch.Tensor) -> torch.Tensor:
#     # Extract the second dimension of the outputs
#     second_dimension = outputs[..., 1]
    
#     # Compute the constraint values (negative if feasible)
#     constraint_values = second_dimension - 2000.0

#     #print(f"Constraint values: {constraint_values}")
    
#     return constraint_values


#bayesian_optimizer = BayesianOptimizer(multiobjective_model=multisingletaskgp, output_constraints=[constraint_func])
#No constraints:
# bayesian_optimizer = BayesianOptimizer(multiobjective_model=multisingletaskgp)

# bayesian_optimizer.optimization_loop(num_iterations=100)

# bayesian_optimizer.visualize_pareto_front()

# bayesian_optimizer.visualize_expected_hypervolume_development()


# sin_x = FunctionFactory


# # TODO this needs to be adjusted here: 

# main_dataset = DataManager(sin_x.multi_inputs)

# main_dataset.load_initial_dataset(num_datapoints=5, bounds=[(0, 6), (0, 2), (0, 3), (0, 2), (2, 3)], maximization_flags=[False, True, True], sampling_method="grid", noise_level=0)

# main_dataset.load_historic_dataset('smart_doe_bayesian_optimization\dataset_creation\pickle_files\datasets.pkl')

# multisingletaskgp = MultiSingletaskGPInitializer(main_dataset)

# multisingletaskgp.initialize_model()