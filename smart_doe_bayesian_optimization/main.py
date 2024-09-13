import models
import torch
import numpy as np
import random
import botorch
from gpytorch.priors import NormalPrior
from data.function_factory import FunctionFactory
from gpytorch.constraints import GreaterThan
import matplotlib.pyplot as plt
from gpytorch.priors.torch_priors import GammaPrior
from utils.config_parser_utils import config_parser
from data.create_dataset import DataManager
from utils.checking_utils import check_type
from models.model_initializer.multi_singletaskgp_initializer import MultiSingletaskGPInitializer  
from optimization.bayesian_optimizer import BayesianOptimizer
from data.constraint_factory import WeldingConstraints
from botorch.test_functions.multi_objective import WeldedBeam
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_mll
from botorch.optim import gen_batch_initial_conditions
from data_export.data_export import export_only_in_out_data
from data.multitask_datasetmanager import MultiTaskDatasetManager
from models.model_initializer.multi_multitask_initialize import MultiMultitaskInitializer
from models.model_initializer.multi_rgpe_initializer import MultiRGPEInitializer
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement

'''
Important bevore running an optimization:
- Check maximization flags
- Check if reference points is needed?
- Check the reference point (negate all values for minimization)
- Resulting Hypervolume is dependent on the outcome ranges! can be very different in size

'''

#setting seeds:
seed = 42
torch.manual_seed(seed=seed)
np.random.seed(seed)
random.seed(seed)
botorch.utils.sampling.manual_seed(seed=seed)

print(50*"-")


# variation_factor = 0.0

# vehicle_crash = FunctionFactory(variation_factor=variation_factor)

# main_dataset = DataManager(external_input=False, 
#                            dataset_func=vehicle_crash.generate_car_crash_synthetic_data, 
#                            historic_data_path="smart_doe_bayesian_optimization\\data_import\\test_data_import", 
#                            variation_factor=variation_factor)

# meta_data_dict = {
#     "var_factor": variation_factor
# }

# main_dataset.load_initial_dataset(num_datapoints=1, bounds=[(1.0, 3.0)] * 5, 
#                                   maximization_flags=[False, False, False], 
#                                   input_parameter_name=["x1", "x2", "x3", "x4", "x5"], 
#                                   output_parameter_name=["Mass", "A_inn", "Intrusion"], meta_data_dict=meta_data_dict, 
#                                   sampling_method="LHS")

# main_dataset.load_historic_data()

# multisingletaskgp = MultiSingletaskGPInitializer(dataset=main_dataset, transfer_learning_method="transfer_and_retrain", bool_transfer_averaging=True)

# multisingletaskgp.initially_setup_model()    

# multisingletaskgp.train_initially_gp_model()

# #reference point handed over as negative values! , reference_point=torch.tensor([-1864.72022, -11.81993945, -0.2903999384], dtype=torch.float64)

# bayesian_optimizer = BayesianOptimizer(multiobjective_model=multisingletaskgp, 
#                                        bool_optional_ending_optimization_each_iteration=False, 
#                                        reference_point=torch.tensor([-1864.72022, -11.81993945, -0.2903999384], dtype=torch.float64), 
#                                        save_file_name="avg_true")

# bayesian_optimizer.optimization_loop(use_stopping_criterion=True, num_max_iterations=70, num_min_iterations=30)


variation_factor = 0.0

vehicle_crash = FunctionFactory(variation_factor=variation_factor)

main_dataset = DataManager(external_input=False, 
                           dataset_func=vehicle_crash.generate_car_crash_synthetic_data, 
                           historic_data_path="smart_doe_bayesian_optimization\\data_import\\test_data_import", 
                           variation_factor=variation_factor)

meta_data_dict = {
    "var_factor": variation_factor
}

main_dataset.load_initial_dataset(num_datapoints=1, bounds=[(1.0, 3.0)] * 5, 
                                  maximization_flags=[False, False, False], 
                                  input_parameter_name=["x1", "x2", "x3", "x4", "x5"], 
                                  output_parameter_name=["Mass", "A_inn", "Intrusion"], meta_data_dict=meta_data_dict, 
                                  sampling_method="LHS")

#main_dataset.load_historic_data()

multisingletaskgp = MultiSingletaskGPInitializer(dataset=main_dataset, transfer_learning_method="no_transfer", bool_transfer_averaging=False)

multisingletaskgp.initially_setup_model()    

multisingletaskgp.train_initially_gp_model()

#reference point handed over as negative values! , reference_point=torch.tensor([-1864.72022, -11.81993945, -0.2903999384], dtype=torch.float64)

bayesian_optimizer = BayesianOptimizer(multiobjective_model=multisingletaskgp, 
                                       bool_optional_ending_optimization_each_iteration=False,  
                                       save_file_name="avg_false")

bayesian_optimizer.optimization_loop(use_stopping_criterion=True, num_max_iterations=70, num_min_iterations=30)

# multi_rgpe = MultiRGPEInitializer(dataset=main_dataset)

# multi_rgpe.setup_model(n_mc_samples=1)

# multi_rgpe.train_initially_gp_model()

# bayesian_optimizer = BayesianOptimizer(multiobjective_model=multi_rgpe, reference_point=torch.tensor([-1864.72022, -11.81993945, -0.2903999384], dtype=torch.float64), save_file_name="var_fac_-0_05")

# bayesian_optimizer.optimization_loop(num_max_iterations=35, num_min_iterations=1)

# variation_factor = 0.5

# vehicle_crash = FunctionFactory(variation_factor=variation_factor)

# main_dataset = DataManager(external_input=False, dataset_func=vehicle_crash.generate_car_crash_synthetic_data, historic_data_path="smart_doe_bayesian_optimization\\data_import\\test_data_import", variation_factor=variation_factor)

# meta_data_dict = {
#     "var_factor": variation_factor
# }

# main_dataset.load_initial_dataset(num_datapoints=15, bounds=[(1.0, 3.0)] * 5, maximization_flags=[False, False, False], input_parameter_name=["x1", "x2", "x3", "x4", "x5"], output_parameter_name=["Mass", "A_inn", "Intrusion"], meta_data_dict=meta_data_dict, sampling_method="LHS")

# main_dataset.load_historic_data()

# print(main_dataset.historic_dataset_list)

# multisingletaskgp = MultiSingletaskGPInitializer(dataset=main_dataset, transfer_learning_method="no_transfer", bool_transfer_averaging=False)

# multisingletaskgp.initially_setup_model()    

# multisingletaskgp.train_initially_gp_model()

# #reference point handed over as negative values! , reference_point=torch.tensor([-1864.72022, -11.81993945, -0.2903999384], dtype=torch.float64)

# bayesian_optimizer = BayesianOptimizer(multiobjective_model=multisingletaskgp, save_file_name="var_fac_-0_05")

# bayesian_optimizer.optimization_loop(num_max_iterations=35, num_min_iterations=1)


#########

# welding_beam = FunctionFactory

# main_dataset = DataManager(dataset_func=welding_beam.welding_beam)

# main_dataset.load_initial_dataset(num_datapoints=5, bounds = [(0.125, 5.0),(0.1, 10.0),(0.1, 10.0),(0.125, 5.0)], maximization_flags=[False, False], sampling_method="grid", noise_level=0)

# multisingletaskgp = MultiSingletaskGPInitializer(main_dataset)

# multisingletaskgp.initially_setup_model()    

# multisingletaskgp.train_initially_gp_model()

# bayesian_optimizer = BayesianOptimizer(multiobjective_model=multisingletaskgp, reference_point=torch.tensor([-40, -0.015], dtype=torch.float64))

# bayesian_optimizer.optimization_loop(num_iterations=100)


#laser_hardening = FunctionFactory

#main_dataset = DataManager(external_input=False, dataset_func=laser_hardening.laser_heat_treatment)

# #main_dataset.load_initial_dataset(num_datapoints=1500, bounds=[(20, 400), (200e-3 / 60, 3000e-3 / 60), (83e-6, 1000e-6)], maximization_flags=[True, False], input_parameter_name=["laser_pwr", "laser_speed", "laser_width"], output_parameter_name=["hardening_time", "temp_div"], sampling_method="grid", noise_level=0)

#main_dataset.load_initial_dataset(num_datapoints=5, bounds=[(20, 400), (200e-3, 3000e-3), (83e-6, 1000e-6)], maximization_flags=[True, False], input_parameter_name=["laser_pwr", "laser_speed", "laser_width"], output_parameter_name=["hardening_time", "temp_div"], sampling_method="grid", noise_level=0)

# export_only_in_out_data(main_dataset.initial_dataset.input_data, main_dataset.initial_dataset.output_data, folder_path="smart_doe_bayesian_optimization\data_export\multi_singletaskgp_data_export", folder_name="TESTTEST_datasets")


#print(main_dataset.initial_dataset.input_data)

#print(main_dataset.initial_dataset.output_data)

# multisingletaskgp = MultiSingletaskGPInitializer(main_dataset)

# multisingletaskgp.initially_setup_model()

# multisingletaskgp.train_initially_gp_model()

# bayesian_optimizer = BayesianOptimizer(multiobjective_model=multisingletaskgp)




