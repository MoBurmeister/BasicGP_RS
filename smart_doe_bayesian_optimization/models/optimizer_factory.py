from torch.optim import Adam, SGD
from typing import Union, Iterable
import torch
from torch.optim.optimizer import ParamsT

# TODO: is the ParamsT type correct for type definition 

class OptimizerFactory:
    """
    Factory class for creating PyTorch optimizers.
    This factory supports the configuration and creation of various types of optimizers based on input specifications,
    allowing for flexible and dynamic instantiation of optimizer objects with custom settings.
    Attributes:
        type (str): The type of the optimizer to be created ('adam', 'SGD', etc.).
        model_parameters (ParamsT): The parameters of the model over which the optimizer will operate.
        **kwargs: Additional keyword arguments specific to the type of optimizer, such as learning rate ('lr'),
                 momentum, etc.
    Methods:
        create_optimizer: Static method to instantiate and return a PyTorch optimizer based on specified parameters.
    """

    @staticmethod
    def create_optimizer(type: str, model_parameters: ParamsT, **kwargs) -> torch.optim.Optimizer:
        """
        Creates and returns an optimizer based on the specified type and associated parameters.
        Parameters:
            type (str): The type of optimizer to create. Supported values are 'adam' and 'SGD'.
            model_parameters (ParamsT): Parameters of the model that the optimizer will update.
            **kwargs: Additional keyword arguments specific to the type of optimizer, such as 'lr' for learning rate,
                      'momentum', and other optimizer-specific settings.
        Returns:
            torch.optim.Optimizer: An instance of a PyTorch optimizer configured according to the specified type
                                   and parameters.
        """
        if type == 'adam':
            return Adam(model_parameters, **kwargs)
        elif type == 'sgd':
            return SGD(model_parameters, **kwargs)
        else:
            raise ValueError(f"Unsupported optimizer type: {type}")