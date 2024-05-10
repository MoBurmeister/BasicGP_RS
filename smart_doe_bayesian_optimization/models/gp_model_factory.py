from botorch.models import SingleTaskGP
from torch import Tensor
from gpytorch.likelihoods import Likelihood
from gpytorch.kernels import Kernel
from typing import Type
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.models.transforms.input import InputTransform, Normalize

# TODO: Add more models like Heteroscedastic
# TODO: Do I need more initilization parameters for the SingleTaskGP? What about mean_module?

class GPModelFactory:
    @staticmethod
    def create_model(model_type: str, train_X: Tensor, train_Y: Tensor, kernel: Kernel, likelihood: Likelihood, outcome_transform: OutcomeTransform, input_transform: InputTransform) -> SingleTaskGP:
        """
        Factory method to create different types of Gaussian Process (GP) models.
        Parameters:
            model_type (str): The type of the GP model to create. Supported models: 'SingleTaskGP'.
            train_X (Tensor): Training input data. Shape should match the model's expected input.
            train_Y (Tensor): Training output data. Shape should match the model's expected output.
            kernel (Kernel): The covariance kernel to be used in the GP model.
            likelihood (Likelihood): The likelihood model to be used with the GP.
        Returns:
            SingleTaskGP: An instance of a Gaussian Process model.
        """
        if model_type == 'SingleTaskGP':
            return SingleTaskGP(train_X=train_X, train_Y=train_Y, likelihood=likelihood, covar_module=kernel, outcome_transform= outcome_transform, input_transform=input_transform)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")