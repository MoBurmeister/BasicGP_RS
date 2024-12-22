from data.create_dataset import DataManager
from models.gp_model import BaseModel
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.module import Module
from gpytorch.means.mean import Mean
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.kernels import MaternKernel
from typing import Literal
import torch
from torch.optim import Adam
import numpy as np

class MultiSingletaskGPInitializer(BaseModel):
    """
    A class to initialize and manage multiple single-task Gaussian Process (GP) models with optional transfer learning.
    Attributes:
        dataset (DataManager): The dataset manager containing initial and historic datasets.
        transfer_learning_method (Literal["no_transfer", "initial_transfer", "transfer_and_retrain"]): The method for transfer learning.
        bool_transfer_averaging (bool): If True, hyperparameters are averaged over all historic datasets.
        lr (float): Learning rate for training.
        step_limit (int): Step limit for training.
    Methods:
        initially_setup_model():
            Sets up the initial GP model. This method should be run only once.
        setup_model():
            Sets up the GP model list based on the initial dataset and applies transfer learning if specified.
        print_model_parameter(model_state_dict):
            Prints the parameters of the GP model for each objective.
        setup_multiple_gp_models():
            Sets up multiple GP models for each objective in the initial dataset.
        train_initially_gp_model():
            Trains the GP model initially after setup.
        reinitialize_model(current_iteration):
            Reinitializes the GP model after new data has been added in each optimization iteration.
        adjust_step_and_lr(current_iteration, lr, step_limit):
            Adjusts the learning rate and step limit based on the current iteration.
        update_old_statedict(old_state_dict):
            Updates the old state dictionary with averaged hyperparameters from historic datasets.
        get_weights_for_transfer():
            Returns the weights for transfer learning based on dataset meta-feature similarity or averaging.
        compute_prior_means_covar_list():
            Computes the prior means and covariance lists for the initial setup from historic datasets.
        extract_mean_covar_from_list(singletaskgp_list_all_tasks):
            Extracts and aggregates mean and covariance modules from a list of single-task GP models.
    """
    def __init__(self, dataset: DataManager, transfer_learning_method: Literal["no_transfer", "initial_transfer", "transfer_and_retrain"], bool_transfer_averaging: bool = True):
        super().__init__(dataset)
        self.gp_model = None
        self.transfer_parameter = None
        #string to clarify transfer learning method
        self.transfer_learning_method = transfer_learning_method
        self.bool_transfer_averaging = bool_transfer_averaging  #if true, the hyperparameters are averaged over all historic datasets
        self.lr = 0.01
        self.step_limit = 100
    
    def initially_setup_model(self): 
        '''
        These models all support multiple outputs. However, as single-task models, SingleTaskGP, FixedNoiseGP, and HeteroskedasticSingleTaskGP 
        should be used only when the outputs are independent and all use the same training data. 
        If outputs are independent and outputs have different training data, use the ModelListGP. When modeling correlations between outputs, 
        use a multi-task model like MultiTaskGP.
        '''
        #this code should be only run once, since it also calculates the parameters

        #historic_gp_model_means, historic_gp_model_covars = self.compute_prior_means_covar_list()

        #self.transfer_parameter = [historic_gp_model_means, historic_gp_model_covars]

        self.setup_model()
        
    
    def setup_model(self):
        """
        Sets up the Gaussian Process (GP) models for the given dataset.
        This method initializes a list of SingleTaskGP models based on the initial dataset provided by the dataset manager.
        If the initial dataset has no data points, no input or output transformations are applied. Otherwise, the input data
        is normalized and the output data is standardized.
        The method also handles transfer learning if specified. If transfer learning is enabled and historic datasets are 
        available, the state dictionary of the initial GP model is updated with parameters from the historic datasets. 
        If no historic datasets are found and transfer learning is required, a ValueError is raised.
        The final GP model list is stored in the `gp_model` attribute of the class.
        Raises:
            ValueError: If transfer learning is enabled but no historic datasets are found.
        Prints:
            - The number of historic datasets found (if transfer learning is enabled).
            - The parameters of the initial GP model before and after updating (if transfer learning is enabled).
            - The parameters of the initial GP model if no transfer learning is applied.
        """

        gp_model_list = []

        #this setups the gp list in a basic fashion

        if self.dataset_manager.initial_dataset.num_datapoints == 0:
            #otherwise error in initalization. When first data is available: Normalize and Standardize the data
            input_transform = None
            output_transform = None
        else:
            input_transform=Normalize(d=self.dataset_manager.initial_dataset.input_dim)
            #m=1 here, since one GP is initialized per objective!
            output_transform=Standardize(m=1)

        for objective in range(self.dataset_manager.initial_dataset.output_dim):        
            gp_model = SingleTaskGP(train_X=self.dataset_manager.initial_dataset.input_data, 
                                     train_Y=self.dataset_manager.initial_dataset.output_data[:, objective].unsqueeze(1),
                                     input_transform=input_transform, 
                                     outcome_transform=output_transform)

            gp_model_list.append(gp_model)

        #star to unpack the modellist here
        gp_modellist = ModelListGP(*gp_model_list)

        if self.transfer_learning_method == "initial_transfer" or self.transfer_learning_method == "transfer_and_retrain":

            if not self.dataset_manager.historic_dataset_list:
                raise ValueError("No historic datasets found. Please provide at least one historic dataset for the Multi_Singletaskmodel!")
            
            print(f"Initializer found {len(self.dataset_manager.historic_datasetinfo_list)} datasets.")

            old_state_dict = gp_modellist.state_dict()

            print("Old Parameters of the initial GP:")
            
            self.print_model_parameter(old_state_dict)

            updated_state_dict = self.update_old_statedict(old_state_dict=old_state_dict)

            gp_modellist.load_state_dict(updated_state_dict)    

            print("Parameter updated successfully. New parameter of the initial GP:")
        
            self.print_model_parameter(gp_modellist.state_dict())

        else:
            if self.transfer_learning_method == "no_transfer":
                print("No historic data available. Using default parameters.")
                self.print_model_parameter(gp_modellist.state_dict())
                print(50*"-")

        self.gp_model = gp_modellist

    def print_model_parameter(self, model_state_dict):
        num_objectives = self.dataset_manager.output_dim
        
        for obj_index in range(num_objectives):
            print(f"Objective {obj_index + 1}:")
            print(f"  constant_mean: {model_state_dict[f'models.{obj_index}.mean_module.raw_constant']}")
            print(f"  raw_lengthscale: {model_state_dict[f'models.{obj_index}.covar_module.base_kernel.raw_lengthscale']}")
            print(f"  raw_outputscale: {model_state_dict[f'models.{obj_index}.covar_module.raw_outputscale']}")
            print(50 * "-")

    def setup_multiple_gp_models(self):
        """
        Sets up multiple Gaussian Process (GP) models for each objective in the dataset.
        This method initializes a list of SingleTaskGP models, one for each objective in the dataset.
        Each GP model is trained on the input data and the corresponding output data for that objective.
        The input data is normalized, and the output data is standardized.
        Returns:
            ModelListGP: A ModelListGP object containing all the initialized SingleTaskGP models.
        """
        gp_model_list = []
        for objective in range(self.dataset_manager.initial_dataset.output_dim):        
            gp_model = SingleTaskGP(train_X=self.dataset_manager.initial_dataset.input_data, 
                                     train_Y=self.dataset_manager.initial_dataset.output_data[:, objective].unsqueeze(1),
                                     input_transform=Normalize(d=self.dataset_manager.initial_dataset.input_dim), 
                                     outcome_transform=Standardize(m=1))
            
            gp_model_list.append(gp_model)
        
        gp_modellist = ModelListGP(*gp_model_list)

        return gp_modellist

    def train_initially_gp_model(self):
        """
        Trains the Gaussian Process (GP) model initially based on the specified transfer learning method.
        This method performs the initial training of the GP model after its setup. Depending on the 
        transfer learning method specified, it either trains the model on the initial dataset or 
        uses historic data for transfer learning.
        Raises:
            ValueError: If no GP model is set.
        Transfer Learning Methods:
            - "no_transfer": No historic data is used. The model is trained on the initial dataset 
              and the Marginal Log Likelihood (MLL) is maximized.
            - "initial_transfer": Historic data is used. The mean and covariance module are fixed 
              initially and will be retrained on new data without maximizing the MLL.
            - "transfer_and_retrain": Similar to "initial_transfer", historic data is used and the 
              model will be retrained on new data without maximizing the MLL.
        """
        #just first initial training after the setup of the model 

        if self.gp_model is None:
            raise ValueError("No GP model set. Please run an initiation first!")
        
        if self.transfer_learning_method == "no_transfer":
            
            print(f"No historic data available. Training on initial dataset with fit_gpytorch_mll. MarginalLogLikelihood is maximized")
            mll = SumMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)

            mll = fit_gpytorch_mll(mll=mll)
    
        elif self.transfer_learning_method == "initial_transfer" or self.transfer_learning_method == "transfer_and_retrain":
            print(f"Historic data is available. Mean and CovModule are taken and fixed right now for initial setup. Will be retrained on new available data. No maximization of MarginalLogLikelihood.")
        
    def reinitialize_model(self, current_iteration: int):
        """
        Reinitializes the model after new data has been added following each optimization iteration.
        Parameters:
        current_iteration (int): The current iteration number of the optimization process.
        Raises:
        ValueError: If no historic data is available and transfer_learning_method is not set to 'no_transfer'.
        The method performs the following steps based on the transfer_learning_method:
        - 'no_transfer': Sets up and trains a new model using only the current dataset.
        - 'initial_transfer': Initializes the model with historic parameters without retraining.
        - 'transfer_and_retrain': Initializes the model with historic parameters and partially retrains it on new data.
        The method also adjusts the learning rate and step limit based on the current iteration when retraining.
        """
        #function to reinitialize the model after new data has been added after each optimization iteration!
        #order: first setup model with data new added data in dataset, then do training

        if len(self.dataset_manager.historic_dataset_list) == 0 and self.transfer_learning_method != "no_transfer":
            raise ValueError("No historic data available. Please set transfer_learning_method to 'no_transfer'.")
        
        print(f"Reinitializing model with {self.dataset_manager.initial_dataset.input_data.shape[0]} data points for iteration {current_iteration}.")

        if self.transfer_learning_method == "no_transfer":
            
            print(f"No historic data available. Model will be refined on the current newly added data!")
            print(f"Fitting the model with {self.dataset_manager.initial_dataset.input_data.shape[0]} data points. GP details: input_train_info: {self.gp_model.train_inputs[0][0].shape}")
            
            gp_modellist = self.setup_multiple_gp_models()

            self.gp_model = gp_modellist

            mll = SumMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)

            mll = fit_gpytorch_mll(mll=mll)

        elif self.transfer_learning_method == "initial_transfer":
            print(f"Initially warm started the model with historic parameters. These will be fixed from now on.")
            #datapoints will be added via condition_on_observations
            #if you did not want to re-train the model hyper-parameters but only update its data, you could use condition_on_observations for that
            #condition_on_observation cannot be used here. This will throw an error later if fit_gpytorch_mll is used
            #this is currently a botorch bug reported here (https://github.com/pytorch/botorch/issues/1435) and not yet resolved
            #so best is to setup a new model an hand over the whole state dict.
            old_state_dict = self.gp_model.state_dict()

            gp_modellist = self.setup_multiple_gp_models()

            gp_modellist.load_state_dict(old_state_dict, strict=False)

            self.gp_model = gp_modellist

            print(f"Old parameters loaded. No retraining")

            self.print_model_parameter(self.gp_model.state_dict())

        elif self.transfer_learning_method == "transfer_and_retrain":
            print(f"Model was initially primed and now it is partially retrained on new data!")

            #setup model with new data and old parameters:

            old_state_dict = self.gp_model.state_dict()

            gp_modellist = self.setup_multiple_gp_models()

            gp_modellist.load_state_dict(old_state_dict, strict=False)

            self.gp_model = gp_modellist

            # torch optimizer method in botorch is used. Optimizer instance is initialized and handed over. Adam is used.
            # see line 182 in botorch/optim/core.py: handed over Optimizer is not overwritten

            mll = SumMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)

            #learning rate and step_limit, which will be increased by schedule
            #these are set as class variables here

            # Adjust learning rate and step limit based on the current iteration
            self.lr, self.step_limit = self.adjust_step_and_lr(current_iteration, self.lr, self.step_limit)

            # Extract all parameters that require gradients from the MLL
            parameters = [p for p in mll.parameters() if p.requires_grad]

            costum_optimizer = Adam(params=parameters, lr=self.lr)

            fit_gpytorch_mll_torch(mll=mll, optimizer=costum_optimizer, step_limit=self.step_limit)

            self.print_model_parameter(self.gp_model.state_dict())
            
    def adjust_step_and_lr(self, current_iteration: int, lr: float, step_limit: int):
        """
        Adjusts the learning rate and step limit based on the current iteration.
        This function doubles the learning rate and step limit every 10 iterations.
        If the current iteration is not a multiple of 10, the learning rate and step limit remain unchanged.
        Args:
            current_iteration (int): The current iteration number.
            lr (float): The current learning rate.
            step_limit (int): The current step limit.
        Returns:
            tuple: A tuple containing the new learning rate and new step limit.
        """
        if current_iteration > 0 and current_iteration % 10 == 0:
            new_lr = lr * 2
            new_step_limit = step_limit * 2
            print(f"Iteration {current_iteration}: Doubling learning rate to {new_lr} and step limit to {new_step_limit}")
        else:
            new_lr = lr
            new_step_limit = step_limit
        
        return new_lr, new_step_limit

    def update_old_statedict(self, old_state_dict: dict):
        """
        Update the old state dictionary with weighted averages of hyperparameters from historic models.
        Args:
            old_state_dict (dict): The state dictionary to be updated.
        Raises:
            ValueError: If the number of weights does not match the number of objectives.
        Returns:
            dict: The updated state dictionary with averaged hyperparameters.
        This method performs the following steps:
        1. Retrieves normalized weights for transfer learning.
        2. Initializes accumulators for each hyperparameter per objective.
        3. Iterates over historic model state dictionaries and extracts weighted hyperparameters.
        4. Computes weighted averages for each hyperparameter per objective.
        5. Prints the averaged hyperparameters for each objective.
        6. Updates the old state dictionary with the averaged values for each objective.
        """

        # Get the normalized weights
        weights = self.get_weights_for_transfer()

        if len(weights) != len(self.dataset_manager.historic_dataset_list):
            raise ValueError("Number of weights must match the number of objectives.")
        
        # Initialize accumulators for each hyperparameter per objective
        constant_means = [[] for _ in range(self.dataset_manager.output_dim)]
        raw_lengthscales = [[] for _ in range(self.dataset_manager.output_dim)]
        raw_outputscales = [[] for _ in range(self.dataset_manager.output_dim)]

        num_objectives = self.dataset_manager.output_dim

        historic_model_statedict_list = self.dataset_manager.historic_modelinfo_list 

        for state_idx, state_dict in enumerate(historic_model_statedict_list):
            for obj_index in range(num_objectives):
                # Extract hyperparameters for each objective and apply weighting
                weighted_constant_mean = weights[state_idx] * state_dict[f'models.{obj_index}.mean_module.raw_constant']
                weighted_lengthscale = weights[state_idx] * state_dict[f'models.{obj_index}.covar_module.base_kernel.raw_lengthscale']
                weighted_outputscale = weights[state_idx] * state_dict[f'models.{obj_index}.covar_module.raw_outputscale']

                constant_means[obj_index].append(weighted_constant_mean)
                raw_lengthscales[obj_index].append(weighted_lengthscale)
                raw_outputscales[obj_index].append(weighted_outputscale)

        # Compute weighted averages for each objective separately
        average_constant_means = [torch.sum(torch.stack(constant_means[obj_index]), dim=0) for obj_index in range(num_objectives)]
        average_raw_lengthscales = [torch.sum(torch.stack(raw_lengthscales[obj_index]), dim=0) for obj_index in range(num_objectives)]
        average_raw_outputscales = [torch.sum(torch.stack(raw_outputscales[obj_index]), dim=0) for obj_index in range(num_objectives)]

        # Print the averaged hyperparameters
        for obj_index in range(num_objectives):
            print(f"Average constant mean for objective {obj_index + 1}: {average_constant_means[obj_index]}")
            print(f"Average raw lengthscale for objective {obj_index + 1}: {average_raw_lengthscales[obj_index]}")
            print(f"Average raw outputscale for objective {obj_index + 1}: {average_raw_outputscales[obj_index]}")

        # Update the old_state_dict with the averaged values for each objective
        for obj_index in range(num_objectives):
            old_state_dict[f'models.{obj_index}.mean_module.raw_constant'].copy_(average_constant_means[obj_index])
            old_state_dict[f'models.{obj_index}.covar_module.base_kernel.raw_lengthscale'].copy_(average_raw_lengthscales[obj_index])
            old_state_dict[f'models.{obj_index}.covar_module.raw_outputscale'].copy_(average_raw_outputscales[obj_index])

        return old_state_dict
    
    def get_weights_for_transfer(self):
        """
        Calculate and return weights for transfer learning based on the specified method.
        This method calculates weights for transfer learning either by averaging or by 
        using dataset meta-feature similarity via Euclidean distance.
        Returns:
            list: A list of weights for each historic dataset.
        Raises:
            ValueError: If meta-features are missing in any historic dataset or if 
                        meta-feature keys are missing in any historic dataset.
        """

        #this should return a number of weights, according to which the hyperparameters are weighted for the transfer learning
        # Number of datasets (or historic models)
        num_datasets = len(self.dataset_manager.historic_dataset_list)

        if self.bool_transfer_averaging:

            print("Averaging weights for transfer learning.")
       
            weights = [1.0 / num_datasets] * num_datasets

        else:

            print("Calculating weights for transfer learning based on dataset meta feature similiarity via euclidian distance.")

            # weighted average for weights according to dataset meta features
            historic_datasetinfo_list = self.dataset_manager.historic_datasetinfo_list
            initial_dataset_metafeatures = self.dataset_manager.initial_dataset.meta_data_dict

            # Extract the meta-features from each dataset and the current task
            historic_metafeatures = []
            for dataset_info in historic_datasetinfo_list:
                if 'metafeatures' not in dataset_info:
                    raise ValueError(f"Metafeatures missing in dataset: {dataset_info}")
                historic_metafeatures.append(dataset_info['metafeatures'])

            # Normalize meta-features for each dataset
            all_metafeatures = historic_metafeatures + [initial_dataset_metafeatures]
            metafeature_keys = list(initial_dataset_metafeatures.keys())
            
            # Check if metafeature keys are present in historic datasets
            for dataset_info in historic_datasetinfo_list:
                if 'metafeatures' not in dataset_info:
                    raise ValueError(f"Metafeatures missing in dataset: {dataset_info}")
                dataset_metafeatures = dataset_info['metafeatures']
                missing_keys = [key for key in metafeature_keys if key not in dataset_metafeatures]
                if missing_keys:
                    raise ValueError(f"Metafeature keys {missing_keys} missing in historic dataset: {dataset_info}")
            
            # Initialize normalization parameters
            metafeature_matrix = np.array([[mf[key] for key in metafeature_keys] for mf in all_metafeatures])
            means = np.mean(metafeature_matrix, axis=0)
            stds = np.std(metafeature_matrix, axis=0)

            # Normalize all meta-features
            normalized_metafeatures = (metafeature_matrix - means) / stds

            # Separate normalized meta-features back
            normalized_historic = normalized_metafeatures[:-1]
            normalized_current = normalized_metafeatures[-1]

            # Calculate Euclidean distances between the current task and each historic dataset
            distances = np.linalg.norm(normalized_historic - normalized_current, axis=1)

            # Convert distances to weights (inverse of distance)
            inverse_distances = 1 / (distances + 1e-8)  # Add a small value to avoid division by zero

            # Normalize weights so they sum to 1
            weights = inverse_distances / np.sum(inverse_distances)

            print(f"Calculated weights for transfer learning: {weights}")

            weights = weights.tolist()

            
        return weights


    #OLD:
               
    #these two functions are only called once in the initial setup, afterwards the state_dict is used to transfer the knowledge!
    def compute_prior_means_covar_list(self) -> tuple[list[Mean], list[Module]]:

        #function for initiation

        singletaskgp_list_all_tasks = []

        #for each historic dataset, train a singletaskgp model per objective of all objectives
        for historic_dataset in self.dataset_manager.historic_modeldata_list:
            
            singletaskgp_list = []

            for objective in range(historic_dataset.output_dim):
                test_var = historic_dataset.output_data[:, objective]
                gp_model = SingleTaskGP(train_X=historic_dataset.input_data,
                                        train_Y=historic_dataset.output_data[:, objective].unsqueeze(1),
                                        input_transform=Normalize(d=historic_dataset.input_dim),
                                        outcome_transform=Standardize(m=1))
                
                mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

                mll = fit_gpytorch_mll(mll=mll)
                #creates a list of GPs per historic dataset
                singletaskgp_list.append(gp_model)
            #overall list takes the lists of GPs of the historic datasets
            singletaskgp_list_all_tasks.append(singletaskgp_list)

        return self.extract_mean_covar_from_list(singletaskgp_list_all_tasks)

    #only once called for initial setup
    def extract_mean_covar_from_list(self, singletaskgp_list_all_tasks: list[list[SingleTaskGP]]) -> tuple[list[Mean], list[Module]]:

        #function for initiation

        mean_list = []
        covar_list = []

        singletaskgp_list_all_tasks = [list(row) for row in zip(*singletaskgp_list_all_tasks)]

        for singletaskgp_list in singletaskgp_list_all_tasks:
            # Extract and aggregate mean modules
            mean = sum([gp.mean_module.constant for gp in singletaskgp_list]) / len(singletaskgp_list)
            mean_list.append(mean)
            
            # Extract and aggregate covariance module parameters
            nu_list = []
            lengthscale_list = []

            for gp in singletaskgp_list:
                if isinstance(gp.covar_module.base_kernel, MaternKernel):
                    nu_list.append(gp.covar_module.base_kernel.nu)
                    '''
                    ARD: Each input dimension gets its own separate lengthscale (i.e. is a non-constant diagonal matrix). 
                    This is controlled by the ard_num_dims keyword argument (as well as has_lengthscale=True).
                    -> The default matern kernel in SingleTaskGP has ard_num_dims=input_dim
                    '''
                    lengthscale_list.append(gp.covar_module.base_kernel.lengthscale)
                else:
                    raise ValueError("Expected Matern kernel as the base kernel.")

            avg_nu = sum(nu_list) / len(nu_list)
            
            #per definition can the nu value only be 1/2, 3/2, or 5/2 for the matern kernel, the closest one ist taken here
            possible_values = [1/2, 3/2, 5/2]
            closest_value = min(possible_values, key=lambda x: abs(x - avg_nu))
            avg_nu = closest_value

            avg_lengthscale = sum(lengthscale_list) / len(lengthscale_list)

            # You may create a new MaternKernel with the aggregated parameters or store the parameters directly
            # For demonstration, let's create a new MaternKernel with the aggregated parameters
            avg_matern_kernel = MaternKernel(nu=avg_nu, ard_num_dims=avg_lengthscale.shape[1])
            avg_matern_kernel.lengthscale = avg_lengthscale 
            
            
            covar_list.append(avg_matern_kernel)

        return mean_list, covar_list  