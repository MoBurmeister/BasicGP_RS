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

        for objective in range(self.dataset_manager.initial_dataset.output_dim):        
            gp_model = SingleTaskGP(train_X=self.dataset_manager.initial_dataset.input_data, 
                                     train_Y=self.dataset_manager.initial_dataset.output_data[:, objective],
                                     input_transform=Normalize(d=self.dataset_manager.initial_dataset.input_dim), 
                                     outcome_transform=Standardize(m=1))

            gp_model.mean_module = self.compute_prior_mean(objective+1)

            gp_model.covar_module = self.compute_prior_covariance(objective+1) 

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

        
               

    def compute_prior_mean(self, objective_dim: int) -> Mean:
        pass

    def compute_prior_covariance(self, objective_dim: int) -> Module:
        pass


    
    
    