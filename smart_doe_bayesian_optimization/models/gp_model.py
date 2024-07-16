# from training import training
# from models.gp_model_factory import GPModelFactory
# from gpytorch.kernels import Kernel
# from models.mll_factory import MLLFactory
# from models.optimizer_factory import OptimizerFactory
# from gpytorch.likelihoods import Likelihood
# from visualization.visualization import GPVisualizer
# from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
# from botorch.fit import fit_gpytorch_mll
# from botorch.models.transforms.outcome import OutcomeTransform, Standardize
# from botorch.models.transforms.input import InputTransform, Normalize
# from utils.checking_utils import check_same_dimension
# import torch
# from typing import Dict, Optional, List, Tuple

from abc import ABC, abstractmethod

from botorch.acquisition.objective import PosteriorTransform
from botorch.posteriors import Posterior
from torch import Tensor
from data.create_dataset import DataManager
from data.dataset import Dataset
from typing import Any, List, Tuple, Union
from botorch.models.model import Model
'''
Shape of the bounds list: 
torch.Size([2, d]) - ([min_x1, min_x2, ...],[max_x1, max_x2, ...])

tensor([[  0,  10, 100],[  6,  20, 200]]) for original input [(0, 6), (10, 20), (100, 200)]
'''

class BaseModel(ABC):
    def __init__(self, dataset_manager: DataManager):
        self.dataset_manager = dataset_manager
       
    @abstractmethod
    def train_initially_gp_model(self):
        pass
    

    def print_model_info(self):
        #print out the model information via print statement in the model class
        pass