from typing import Union
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO

class MLLFactory:
    @staticmethod
    def create_mll(type: str, model: Union[SingleTaskGP], likelihood: Likelihood) -> Union[ExactMarginalLogLikelihood, VariationalELBO]:
        """
        Creates and returns an MLL object based on specified type and associated with given model and likelihood.
        Parameters:
            type (str): Type of MLL to create. This specifies the configuration and calculation method of the MLL.
            model (GaussianProcessModel): The GP model for which the MLL is computed. This is typically an instance of a class from gpytorch.models.
            likelihood (Likelihood): The likelihood associated with the GP model. This is generally an instance of a class from gpytorch.likelihoods.
        Returns:
            MarginalLogLikelihood: An instance of Marginal Log Likelihood suitable for the GP model.
        """
        if type == 'ExactMarginalLogLikelihood':
            return ExactMarginalLogLikelihood(likelihood, model)
        elif type == 'VariationalELBO':
            return VariationalELBO(likelihood, model, num_data=len(model.train_targets))
        else:
            raise ValueError(f"Unsupported MLL type: {type}")