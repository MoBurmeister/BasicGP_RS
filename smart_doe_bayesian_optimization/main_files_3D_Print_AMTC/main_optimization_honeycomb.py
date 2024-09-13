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


variation_factor = 0.0


main_dataset = DataManager(external_input=True, 
                           historic_data_path="smart_doe_bayesian_optimization\\data_import\\test_data_import", 
                           variation_factor=variation_factor)

#this setup is important and needs to be adjusted
meta_data_dict = {
    "complexity": 7,
    "print_distance": 0.43,
    "infill_weight": 1.30
}

#objectives: y1: minimize cycle time / y2: maximize bending force

main_dataset.load_initial_dataset(num_datapoints=1, 
                                  bounds=[(0.15, 0.45), (75, 300)], 
                                  maximization_flags=[False, True], 
                                  input_parameter_name=["infill_percentage", "printing_speed"], 
                                  output_parameter_name=["cycle_time", "bending_force"], 
                                  meta_data_dict=meta_data_dict, 
                                  sampling_method="LHS")

multisingletaskgp = MultiSingletaskGPInitializer(dataset=main_dataset, transfer_learning_method="no_transfer", bool_transfer_averaging=True)

multisingletaskgp.initially_setup_model()    

multisingletaskgp.train_initially_gp_model()

#reference point handed over as negative values!

bayesian_optimizer = BayesianOptimizer(multiobjective_model=multisingletaskgp, 
                                       bool_optional_ending_optimization_each_iteration=True, 
                                       save_file_name="TL_opt_honeycomb")

bayesian_optimizer.optimization_loop(use_stopping_criterion=False, num_max_iterations=45, num_min_iterations=15)



