from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound 
from gpytorch.models import ExactGP
import torch

# TODO: is a transform in the acquisition function required? Excerpt from Expected Improvement:
'''
posterior_transform: A PosteriorTransform. If using a multi-output model,
a PosteriorTransform that transforms the multi-output posterior into a
single-output posterior is required.
'''

# TODO: Implementation of a parameter xi to adapt exploration/exploitation, can be implemented as a substraction from the mean (e.g. in EI) -> see https://github.com/pytorch/botorch/issues/373

class AcquisitionFunctionFactory():

    @staticmethod
    def create_acquisition_function(acq_function_type: str, gp_model: ExactGP, train_Y: torch.Tensor, maximization: bool = True, **kwargs):

        """
        Factory method to create different types of acquisition functions used in Bayesian Optimization.

        Parameters:
        - acq_function_type (str): The type of acquisition function to create. Possible: 'Exp_Improvement', 'LogExp_Improvement', 'Prob_Improvement', 'Up_Conf_Bound'
        - gp_model (ExactGP): Gaussian Process model used by the acquisition function.
        - train_Y (torch.Tensor): The outputs from the training data.
        - maximization (bool): Flag to indicate if the objective is maximization. Default is True.
        - kwargs: Additional keyword arguments for specific acquisition functions. In this case only for the UpperConfidenceBound Acquisition Function

        Returns:
        - An instance of the specified acquisition function.

        Raises:
        - ValueError: If an unsupported acquisition function type is provided.
        """
        
        if maximization:
            best_f = train_Y.max()
        else:
            best_f = train_Y.min()

        if acq_function_type == 'Exp_Improvement':
            return ExpectedImprovement(model = gp_model, best_f=best_f, maximize=maximization)
        elif acq_function_type == 'LogExp_Improvement':
            return LogExpectedImprovement(model=gp_model, best_f=best_f, maximize=maximization)
        elif acq_function_type == 'Prob_Improvement':
            return ProbabilityOfImprovement(model=gp_model, best_f=best_f, maximize=maximization)
        elif acq_function_type == 'Up_Conf_Bound':
            return UpperConfidenceBound(model=gp_model, maximize=maximization, **kwargs)
        else:
            raise ValueError(f"Unsupported acquisition function type: {acq_function_type}")