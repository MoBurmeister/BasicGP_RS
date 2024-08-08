from botorch.acquisition.acquisition import AcquisitionFunction
from data.create_dataset import DataManager
from models.gp_model import BaseModel
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.module import Module
from gpytorch.means.mean import Mean
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.kernels import MaternKernel
from typing import Literal
import torch

class MultiSingletaskGPInitializer(BaseModel):
    def __init__(self, dataset: DataManager, transfer_learning_method: Literal["no_transfer", "initial_transfer", "transfer_and_retrain"]):
        super().__init__(dataset)
        self.gp_model = None
        self.transfer_parameter = None
        #string to clarify transfer learning method
        self.transfer_learning_method = transfer_learning_method
    
    def initially_setup_model(self): 
        '''
        These models all support multiple outputs. However, as single-task models, SingleTaskGP, FixedNoiseGP, and HeteroskedasticSingleTaskGP 
        should be used only when the outputs are independent and all use the same training data. 
        If outputs are independent and outputs have different training data, use the ModelListGP. When modeling correlations between outputs, 
        use a multi-task model like MultiTaskGP.
        '''
        #this code should be only run once, since it also calculates the parameters

        historic_gp_model_means, historic_gp_model_covars = self.compute_prior_means_covar_list()

        self.transfer_parameter = [historic_gp_model_means, historic_gp_model_covars]

        self.setup_model()
        
    
    def setup_model(self):
        # TODO: Implement retraining function for initialization
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

        # TODO: additional Parameter setup needed here?
        for objective in range(self.dataset_manager.initial_dataset.output_dim):        
            gp_model = SingleTaskGP(train_X=self.dataset_manager.initial_dataset.input_data, 
                                     train_Y=self.dataset_manager.initial_dataset.output_data[:, objective].unsqueeze(1),
                                     input_transform=input_transform, 
                                     outcome_transform=output_transform)

            if self.transfer_learning_method == "initial_transfer" or self.transfer_learning_method == "transfer_and_retrain":
                print(f"Objective {objective + 1}:")
                print("Old Parameters:")
                print(gp_model.mean_module.constant.item())
                print(gp_model.covar_module.base_kernel.lengthscale.tolist())
                print(50*"-")

            historic_gp_model_means, historic_gp_model_covars = self.transfer_parameter

            if historic_gp_model_covars or historic_gp_model_means:

                if historic_gp_model_covars is None or historic_gp_model_means is None:
                    raise ValueError("Both historic_gp_model_covars and historic_gp_model_means must be set.")
            
                gp_model.mean_module.constant = historic_gp_model_means[objective]
                gp_model.covar_module = historic_gp_model_covars[objective]
                
                print("New Parameters:")
                print(gp_model.mean_module.constant.item())
                print(gp_model.covar_module.lengthscale.tolist())
                print(50*"-")

            else:
                if self.transfer_learning_method == "no_transfer":
                    print("No historic data available. Using default parameters.")
                    print(50*"-")

            gp_model_list.append(gp_model)

        #star to unpack the modellist here
        gp_modellist = ModelListGP(*gp_model_list)

        self.gp_model = gp_modellist

    def setup_multiple_gp_models(self):
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
        #just first initial training after the setup of the model 

        if self.gp_model is None:
            raise ValueError("No GP model set. Please run an initiation first!")
        
        if self.transfer_learning_method == "no_transfer":
            
            print(f"No historic data available. Training on initial dataset with fit_gpytorch_mll. MarginalLogLikelihood is maximized")
            mll = SumMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)

            mll = fit_gpytorch_mll(mll=mll)
    
        elif self.transfer_learning_method == "initial_transfer":
            print(f"Historic data is available. Mean and CovModule are taken and fixed right now. No maximization of MarginalLogLikelihood.")
        elif self.transfer_learning_method == "transfer_and_retrain":
            print(f"Historic data is available. Model was initially primed and now it is partially retrained on new data! afterwards")
        
    def reinitialize_model(self, initial_train_method: str = "no_retrain"):
        #function to reinitialize the model after new data has been added after each optimization iteration!

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

            gp_modellist.load_state_dict(old_state_dict)

            self.gp_model = gp_modellist

            print(f"Old parameters loaded. No retraining")

            
        # TODO: retraining scheduler needs to be implemented here.

        elif self.transfer_learning_method == "transfer_and_retrain":
            raise NotImplementedError("This method is not yet implemented.")        
            
               
    #these two functions are only called once in the initial setup, afterwards the state_dict is used to transfer the knowledge!
    def compute_prior_means_covar_list(self) -> tuple[list[Mean], list[Module]]:

        #function for initiation

        singletaskgp_list_all_tasks = []

        #for each historic dataset, train a singletaskgp model per objective of all objectives
        for historic_dataset in self.dataset_manager.historic_datasets:
            
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