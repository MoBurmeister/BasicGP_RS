import torch
from data.function_factory import FunctionFactory
from data.create_dataset import DataManager
from models.model_initializer.multi_singletaskgp_initializer import MultiSingletaskGPInitializer
from models.model_initializer.multi_multitask_initialize import MultiMultitaskInitializer
from models.model_initializer.multi_rgpe_initializer import MultiRGPEInitializer
from optimization.bayesian_optimizer import BayesianOptimizer
print(50*"-")

variation_factors = [-0.05, -0.1, 0.05, 0.1]

num_data_points_initial_dataset = 10
num_max_iterations = 70
num_min_iterations = 35

for variation_factor in variation_factors:
    print(50*"-")

    vehicle_crash = FunctionFactory(variation_factor=variation_factor)

    main_dataset = DataManager(external_input=False, dataset_func=vehicle_crash.generate_car_crash_synthetic_data, variation_factor=variation_factor)

    meta_data_dict = {
        "var_factor": variation_factor
    }

    main_dataset.load_initial_dataset(num_datapoints=num_data_points_initial_dataset, 
                                      bounds=[(1.0, 3.0)] * 5, 
                                      maximization_flags=[False, False, False], 
                                      input_parameter_name=["x1", "x2", "x3", "x4", "x5"], 
                                      output_parameter_name=["Mass", "A_inn", "Intrusion"], 
                                      meta_data_dict=meta_data_dict, 
                                      sampling_method="LHS")
    
    multi_singletaskgp = MultiSingletaskGPInitializer(dataset=main_dataset, transfer_learning_method="no_transfer", bool_transfer_averaging=False)

    multi_singletaskgp.initially_setup_model()

    multi_singletaskgp.train_initially_gp_model()

    bayesian_optimizer = BayesianOptimizer(multiobjective_model=multi_singletaskgp,
                                           bool_optional_ending_optimization_each_iteration=False,
                                           save_file_name=f"SECOND_{variation_factor}")
    
    bayesian_optimizer.optimization_loop(use_stopping_criterion=True,
                                         num_max_iterations=num_max_iterations, 
                                         num_min_iterations=num_min_iterations)
                                           