from data.create_dataset import DataManager
from models.gp_model import BaseModel
from data.multitask_datasetmanager import MultiTaskDatasetManager
from botorch.models import MultiTaskGP
from botorch.models import ModelList
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

class MultiMultitaskInitializer(BaseModel):
    def __init__(self, dataset_manager: DataManager):
        super().__init__(dataset_manager)

        if dataset_manager.initial_dataset.input_data is None:
            raise ValueError("The initial dataset cannot be empty for the MultiMultitaskInitializer!")

        self.gp_model = None
        self.multitaskdatasetmanager = MultiTaskDatasetManager(self.dataset_manager)

    
    def initially_setup_model(self):

        if not self.dataset_manager.historic_dataset_list:
            raise ValueError("No historic datasets found. Please provide at least one historic dataset for the MultiMultitaskInitializer!")

        print(f"Number of historic datasets: {len(self.dataset_manager.historic_dataset_list)}. Can instantiate MultiMultitaskInitializer")

        if self.dataset_manager.initial_dataset.output_dim == 0:
            raise ValueError("No output dimension found. Please provide at least one output dimension for the MultiMultitaskInitializer!")

        multitaskmodel_list = []

        #Task feature is at index 0 of the input data with task index 0 being the initial dataset and upcounting for all historic datasets

        #transformations: 
        #add 1 here for the task feature dimension
        input_transform = Normalize(d=self.dataset_manager.input_dim+1)
        #m=1 since one objective per multitaskGP
        output_transform = Standardize(m=1)

        for objective in range(self.dataset_manager.initial_dataset.output_dim):  
            multitaskmodel = MultiTaskGP(train_X=self.multitaskdatasetmanager.train_X_taskdataset, 
                                         train_Y=self.multitaskdatasetmanager.train_Y_single_taskdatasets[objective], 
                                         input_transform=input_transform,
                                         outcome_transform=output_transform,
                                         task_feature=0,
                                         output_tasks=[0]
                                         )
            
            multitaskmodel_list.append(multitaskmodel)

        multitask_modellist = ModelList(*multitaskmodel_list)

        self.gp_model = multitask_modellist

        print(f"MultiMultitaskInitializer model successfully initialized. Number of multitask models: {self.gp_model.num_outputs} for {self.dataset_manager.initial_dataset.output_dim} objectives.")

    def train_initially_gp_model(self):

        if self.gp_model is None:
            raise ValueError("No GP model set. Please run an initiation first!")
        
        #the mlls of all single multitask models must be trained individually

        for i in range(self.gp_model.num_outputs):
            mll = ExactMarginalLogLikelihood(self.gp_model.models[i].likelihood, self.gp_model.models[i])

            mll = fit_gpytorch_mll(mll)

        print("MultiMultitaskInitializer model successfully trained.")

    def reinitialize_model(self, current_iteration: int):

        print(f"Reinitializing model with {self.dataset_manager.initial_dataset.input_data.shape[0]} data points for iteration {current_iteration}.")

        multitaskmodel_list = []

        #transformations: 
        #add 1 here for the task feature dimension
        input_transform = Normalize(d=self.dataset_manager.input_dim+1)
        #m=1 since one objective per multitaskGP
        output_transform = Standardize(m=1)

        for objective in range(self.dataset_manager.initial_dataset.output_dim):  
            multitaskmodel = MultiTaskGP(train_X=self.multitaskdatasetmanager.train_X_taskdataset, 
                                         train_Y=self.multitaskdatasetmanager.train_Y_single_taskdatasets[objective], 
                                         input_transform=input_transform,
                                         outcome_transform=output_transform,
                                         task_feature=0,
                                         output_tasks=[0]
                                         )
            
            multitaskmodel_list.append(multitaskmodel)

        multitask_modellist = ModelList(*multitaskmodel_list)

        self.gp_model = multitask_modellist

        for i in range(self.gp_model.num_outputs):
            mll = ExactMarginalLogLikelihood(self.gp_model.models[i].likelihood, self.gp_model.models[i])
            mll = fit_gpytorch_mll(mll)

        for i in range(self.gp_model.num_outputs):
            print(f"Multitaskmodel initialized with {self.gp_model.models[i].train_inputs[0].shape} input datapoints and {self.gp_model.models[i].train_targets.shape} output points.")

        print(f"MultiMultitaskInitializer model successfully reinitialized and retrained.")



        

            








