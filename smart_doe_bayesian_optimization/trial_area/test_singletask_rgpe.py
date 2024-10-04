import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import GP
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import PsdSumLazyTensor
from gpytorch.likelihoods import LikelihoodList
from torch.nn import ModuleList
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement

def generate_y(train_X, variation_factor):
    """
    Generate output y based on the input train_X with a variation factor.

    Args:
        train_X (Tensor): Input tensor of shape (n, d) where n is the number of data points
                          and d is the dimension of each data point (here, d=2).
        variation_factor (float): A factor that introduces variation into the output.

    Returns:
        Tensor: Output tensor y of shape (n, 1).
    """
    x1 = train_X[:, 0]
    x2 = train_X[:, 1]
    
    # Sinusoidal function with variation factor applied
    y = torch.sin(x1 * (2 * torch.pi) * variation_factor) + torch.cos(x2 * (2 * torch.pi)) * (1 + variation_factor)
    
    return y.unsqueeze(-1)

# Generate sample two-dimensional training data
train_X = torch.rand(10, 2, dtype=torch.float64)  # 10 training points, each with 2 dimensions

# Variation factors for different models
variation_factors = [1.0, 2.0, 0.5]

# Generate train_Y using the single function with different variation factors
train_Y_1 = generate_y(train_X, variation_factors[0])
train_Y_2 = generate_y(train_X, variation_factors[1])
train_Y_3 = generate_y(train_X, variation_factors[2])

input_transform = Normalize(d=2)

output_transform_1 = Standardize(m=1)

# Initialize the SingleTaskGP models
model_1 = SingleTaskGP(train_X, train_Y_1, input_transform=input_transform, outcome_transform=output_transform_1)
model_2 = SingleTaskGP(train_X, train_Y_2, input_transform=input_transform, outcome_transform=output_transform_1)
model_3 = SingleTaskGP(train_X, train_Y_3, input_transform=input_transform, outcome_transform=output_transform_1)

# Fit the models
mll_1 = ExactMarginalLogLikelihood(model_1.likelihood, model_1)
mll_2 = ExactMarginalLogLikelihood(model_2.likelihood, model_2)
mll_3 = ExactMarginalLogLikelihood(model_3.likelihood, model_3)

fit_gpytorch_model(mll_1)
fit_gpytorch_model(mll_2)
fit_gpytorch_model(mll_3)

# Define bounds for the input space
bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)  

modellist = [model_1, model_2, model_3]

weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)  # Weights for the models

class RGPE(GP, GPyTorchModel):
    """
    Rank-weighted GP ensemble. Note: this class inherits from GPyTorchModel which provides an
        interface for GPyTorch models in botorch.
    """

    _num_outputs = 1  # metadata for botorch

    def __init__(self, models, weights):
        super().__init__()
        self.models = ModuleList(models)
        for m in models:
            if not hasattr(m, "likelihood"):
                raise ValueError(
                    "RGPE currently only supports models that have a likelihood (e.g. ExactGPs)"
                )
        self.likelihood = LikelihoodList(*[m.likelihood for m in models])
        self.weights = weights
        self.to(weights)

    def forward(self, x):
        weighted_means = []
        weighted_covars = []
        # filter model with zero weights
        # weights on covariance matrices are weight**2
        non_zero_weight_indices = (self.weights**2 > 0).nonzero()
        non_zero_weights = self.weights[non_zero_weight_indices]
        # re-normalize
        non_zero_weights /= non_zero_weights.sum()

        for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
            raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
            model = self.models[raw_idx]
            posterior = model.posterior(x)
            # unstandardize predictions
            posterior_mean = posterior.mean.squeeze(-1) * model.outcome_transform.stdvs + model.outcome_transform.means
            posterior_cov = posterior.mvn.lazy_covariance_matrix * model.outcome_transform.stdvs.pow(2)
            # apply weight
            weight = non_zero_weights[non_zero_weight_idx]
            weighted_means.append(weight * posterior_mean)
            weighted_covars.append(posterior_cov * weight**2)
        # set mean and covariance to be the rank-weighted sum the means and covariances of the
        # base models and target model
        mean_x = torch.stack(weighted_means).sum(dim=0)
        covar_x = PsdSumLazyTensor(*weighted_covars)
        return MultivariateNormal(mean_x, covar_x)

# This part here would be adjusted to catch just the SingleTaskGP models for the considered objective each
rgpe = RGPE(modellist, weights)

rgpe_2 = RGPE(modellist, weights)

list_of_models = [rgpe, rgpe_2]

modellist = ModelListGP(*list_of_models)

# Define the Noisy Expected Hypervolume Improvement (NEHVI) acquisition function

acq_function = qLogNoisyExpectedHypervolumeImprovement(model=modellist, 
                                                       ref_point=torch.tensor([0.0, 0.0], dtype=torch.float64),
                                                       X_baseline=train_X,
                                                       prune_baseline=False)

candidate, acq_value = optimize_acqf(
            acq_function=acq_function,
            bounds=bounds,
            q=1,
            num_restarts=40,
            raw_samples=512
        )


print(f"Optimal solution: {candidate}")