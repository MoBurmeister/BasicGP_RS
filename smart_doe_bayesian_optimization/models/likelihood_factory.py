import gpytorch.likelihoods as likelihoods
import gpytorch.constraints as constraints
from gpytorch.constraints import GreaterThan

class LikelihoodFactory:
    @staticmethod
    def create_likelihood(likelihood_dict: dict) -> likelihoods.Likelihood:
        
        likelihood_type = likelihood_dict['likelihood_type']
        likelihood_params = parse_likelihood_params(likelihood_dict['likelihood_params'])
        
        if likelihood_type == 'Gaussian':
            return likelihoods.GaussianLikelihood(**likelihood_params)
        elif likelihood_type == 'Bernoulli':
            return likelihoods.BernoulliLikelihood(**likelihood_params)
        elif likelihood_type == 'Multitask':
            return likelihoods.MultitaskGaussianLikelihood(**likelihood_params)
        else:
            raise ValueError(f"Unknown likelihood type: {likelihood_type}")
        
def parse_likelihood_params(likelihood_params):
    parsed_params = {}
    for key, value in likelihood_params.items():
        if key == 'noise_constraints':
            constraint_type = value['noise_constraints_type']
            # TODO: add other necessary prior types or variable types for handover
            if constraint_type == 'GreaterThan':
                parsed_params['noise_constraint'] = GreaterThan(value['min_val'])
        else:
            parsed_params[key] = value
    return parsed_params