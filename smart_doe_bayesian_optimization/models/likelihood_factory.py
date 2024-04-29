import gpytorch.likelihoods as likelihoods
import gpytorch.constraints as constraints

class LikelihoodFactory:
    @staticmethod
    def create_likelihood(likelihood_type: str, **kwargs):
        """
        Creates a likelihood based on the specified type and parameters.
        
        Args:
            likelihood_type (str): The type of likelihood to create. Examples include 'Gaussian', 'Bernoulli', 'Multitask'.
            **kwargs: Arbitrary keyword arguments for configuring the likelihood.
        
        Returns:
            gpytorch.likelihoods.Likelihood: The instantiated likelihood object.
        
        Raises:
            ValueError: If the likelihood type is unknown or if required parameters are missing.
        """
        if likelihood_type == 'Gaussian':
            return likelihoods.GaussianLikelihood(**kwargs)
        elif likelihood_type == 'Bernoulli':
            return likelihoods.BernoulliLikelihood(**kwargs)
        elif likelihood_type == 'Multitask':
            return likelihoods.MultitaskGaussianLikelihood(**kwargs)
        else:
            raise ValueError(f"Unknown likelihood type: {likelihood_type}")