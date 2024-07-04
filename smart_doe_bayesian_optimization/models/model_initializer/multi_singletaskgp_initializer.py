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
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel

class MultiSingletaskGPInitializer(BaseModel):
    def __init__(self, dataset: DataManager):
        super().__init__(dataset)
        self.gp_model = None
        self.mll = None
    
    def initialize_model(self): 
        '''
        These models all support multiple outputs. However, as single-task models, SingleTaskGP, FixedNoiseGP, and HeteroskedasticSingleTaskGP 
        should be used only when the outputs are independent and all use the same training data. 
        If outputs are independent and outputs have different training data, use the ModelListGP. When modeling correlations between outputs, 
        use a multi-task model like MultiTaskGP.
        '''

        gp_model_list = []

        historic_gp_model_means, historic_gp_model_covars = self.compute_prior_means_covar_list()

        for objective in range(self.dataset_manager.initial_dataset.output_dim):        
            gp_model = SingleTaskGP(train_X=self.dataset_manager.initial_dataset.input_data, 
                                     train_Y=self.dataset_manager.initial_dataset.output_data[:, objective],
                                     input_transform=Normalize(d=self.dataset_manager.initial_dataset.input_dim), 
                                     outcome_transform=Standardize(m=1))

            gp_model.mean_module = historic_gp_model_means[objective]

            gp_model.covar_module = historic_gp_model_covars[objective]

            gp_model_list.append(gp_model)

        #star to unpack the modellist here
        gp_modellist = ModelListGP(*gp_model_list)

        self.gp_model = gp_modellist

    def train_gp_model(self):
        #just first initial training 

        if self.gp_model is None:
            raise ValueError("No GP model set. Please run an initiation loop first!")
        
        mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)

        mll = fit_gpytorch_mll(mll=mll)
               
    #these two functions are only called once in the initial setup, afterwards the state_dict is used to transfer the knowledge!
    def compute_prior_means_covar_list(self) -> tuple[list[Mean], list[Module]]:

        #function for initiation

        singletaskgp_list_all_tasks = []

        for historic_dataset in self.dataset_manager.historic_datasets:
            
            singletaskgp_list = []

            for objective in range(historic_dataset.output_dim):
                test_var = historic_dataset.output_data[:, objective]
                gp_model = SingleTaskGP(train_X=historic_dataset.input_data,
                                        train_Y=historic_dataset.output_data[:, objective].unsqueeze(1),
                                        input_transform=Normalize(d=historic_dataset.input_dim),
                                        outcome_transform=Standardize(m=1))
                
                singletaskgp_list.append(gp_model)
            
            singletaskgp_list_all_tasks.append(singletaskgp_list)

        return self.extract_mean_covar_from_list(singletaskgp_list_all_tasks)

    
    def extract_mean_covar_from_list(self, singletaskgp_list_all_tasks: list[list[SingleTaskGP]]) -> tuple[list[Mean], list[Module]]:

        #function for initiation

        mean_list = []
        covar_list = []

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

        
            
        


    
    
    