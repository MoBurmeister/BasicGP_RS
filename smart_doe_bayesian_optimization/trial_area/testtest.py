#imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time  
import botorch
from botorch.models import MultiTaskGP, SingleTaskGP, ModelList
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.models.transforms.input import InputTransform, Normalize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model, fit_gpytorch_mll
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
import torch
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import GP
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.lazy import PsdSumLazyTensor
from gpytorch.likelihoods import LikelihoodList
from torch.nn import ModuleList
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from linear_operator.operators import BlockDiagLinearOperator, DenseLinearOperator

#data generation: 


#data generation: 
 
from botorch.test_functions.multi_objective import VehicleSafety

bounds = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0, 3.0]])

train_X = torch.rand(4, 5, dtype=torch.float64) * (bounds[1] - bounds[0]) + bounds[0]
function = VehicleSafety()


def evaluate_train_y(x):
    # Evaluate the train_Y values for the given x tensor
    y = function.evaluate_true(x)
    
    y = -y
    
    return y

train_Y = evaluate_train_y(train_X)   

print(train_X, train_Y)

train_X = train_X.to(dtype=torch.float64)
train_Y = train_Y*-1
train_Y = train_Y.to(dtype=torch.float64)

num_objectives = train_Y.shape[1]

modelllist_gp = []

for objective in range(num_objectives):
    gp_model = SingleTaskGP(train_X=train_X, 
                 train_Y=train_Y[:, objective].unsqueeze(-1),
                 input_transform=Normalize(d=5), 
                 outcome_transform=Standardize(m=1))
    
    modelllist_gp.append(gp_model)

gp_modellist_1 = ModelList(*modelllist_gp)

mll = SumMarginalLogLikelihood(likelihood=gp_modellist_1.likelihood, model=gp_modellist_1) 

mll = fit_gpytorch_model(mll)


modelllist_gp_2 = []

for objective in range(num_objectives):
    gp_model = SingleTaskGP(train_X=train_X, 
                 train_Y=train_Y[:, objective].unsqueeze(-1),
                 input_transform=Normalize(d=5), 
                 outcome_transform=Standardize(m=1))
    
    modelllist_gp_2.append(gp_model)

gp_modellist_2 = ModelList(*modelllist_gp_2)

mll_2 = SumMarginalLogLikelihood(likelihood=gp_modellist_2.likelihood, model=gp_modellist_2) 

mll_2 = fit_gpytorch_model(mll_2)


class RGPE(GP, GPyTorchModel):

    def __init__(self, models, weights, num_outputs):
        super().__init__()
        self.models = ModuleList(models)
        for m in models:
            if not hasattr(m, "likelihood"):
                raise ValueError(
                    "RGPE currently only supports models that have a likelihood (e.g. ExactGPs)"
                )
        self.likelihood = LikelihoodList(*[m.likelihood for m in models])
        self.weights = weights
        self._num_outputs = num_outputs
        self.to(weights)

    def forward(self, x):
        # Compute the weighted mean and covariance
        weighted_means = []
        weighted_covariances = []
        
        for i, model in enumerate(self.models):
            posterior = model.posterior(x)
            mean_x = posterior.mean
            covar_x = posterior.mvn.lazy_covariance_matrix

            #print(f"covar_x {i} dense: {covar_x.to_dense()}\n")
            
            #unstandardizing
            mean_x = mean_x.squeeze(-1) * model.outcome_transform.stdvs + model.outcome_transform.means
            covar_x = covar_x
            
            # Compute weighted mean and covariance
            weighted_means.append(self.weights[i] * mean_x)
            weighted_covariances.append(self.weights[i]**2 * covar_x)

            #print(f"Weighted Covariance {i} (Dense): \n{weighted_covariances[i].to_dense()}\n")

        #Sum the weighted means and covariances
        mean_x = sum(weighted_means)
        covar_x = weighted_covariances[0]  # Start with the first covariance
    
        #Sum the weighted covariances - all are BlockDiagLinearOperator
        base_covariances = [covar.base_linear_op for covar in weighted_covariances]

        summed_covariance = BlockDiagLinearOperator(sum(base_covariances))

        #print(f"Summed Covariance (Dense): \n{summed_covariance.to_dense()}\n")


        #Return MultivariateNormal with weighted mean and covariance
        return MultitaskMultivariateNormal(mean_x, summed_covariance)


rgpe = RGPE(models=[model, model_2], weights=torch.tensor([0.5, 0.5], dtype=torch.float64), num_outputs=3)

rgpe_posterior = rgpe.posterior(train_X)



ref_point = torch.tensor([-1864.72022, -11.81993945, -0.2903999384], dtype=torch.float64)  #Three objectives and ref point from VehicleSafety


print("X_baseline shape:", train_X.shape)
print("train_Y shape:", train_Y.shape)
print("train_X shape:", train_X.shape)
print("ref_point shape:", ref_point.shape)
print("model output shape:", rgpe.posterior(train_X).mean.shape) 
print(f"model_1 output shape: {model.posterior(train_X).mean.shape}")

acq_function = qLogNoisyExpectedHypervolumeImprovement(model=rgpe, 
                                                            ref_point=ref_point, 
                                                            X_baseline=train_X,
                                                            prune_baseline=False)
        

candidate, acq_value = optimize_acqf(
acq_function=acq_function,
bounds=bounds,
q=1,
num_restarts=40,
raw_samples=512
        )

print(candidate, acq_value)

# def objective_functions(tensor):
#     x1, x2, x3, x4 = tensor[0]
#     f1 = 4.9e-5 * (x2**2 - x1**2) * (x4 - 1)
#     f2 = 9.82e6 * (x2**2 - x1**2) / (x3 * x4 * (x2**3 - x1**3))
#     return torch.tensor([[-f1, -f2]])  # Negate the objectives for minimization

# #bounds definition:
# parameter_bounds = torch.tensor([[  55.,   75., 1000.,    2.], [  80.,  110., 3000.,   20.]], dtype=torch.float64)

# def get_initial_dataset(bounds: torch.Tensor, num_points: int = 10):
#     """
#     Generates an initial dataset for the SingleTaskGP.

#     Parameters:
#     bounds (torch.Tensor): A 2 x d tensor specifying the lower and upper bounds for each dimension.
#     num_points (int): Number of points to sample. Default is 10.

#     Returns:
#     train_X (torch.Tensor): A n x d tensor of training features.
#     train_Y (torch.Tensor): A n x m tensor of training observations.
#     """
#     # Ensure the bounds are of float type
#     bounds = bounds.float()

#     # Create a grid of points within the bounds
#     dim = bounds.size(1)
#     train_X = torch.zeros(num_points, dim)

#     # Calculate the step size for each dimension
#     steps = (bounds[1] - bounds[0]) / (num_points - 1)

#     # Generate points within the bounds
#     for i in range(num_points):
#         for j in range(dim):
#             train_X[i, j] = bounds[0, j] + i * steps[j]

#     # Evaluate the objective functions for each point
#     train_Y = torch.zeros(num_points, 2)  # Assuming there are 2 objective functions
#     for i in range(num_points):
#         train_Y[i] = objective_functions(train_X[i].unsqueeze(0))

#     train_X = train_X.to(dtype=torch.float64)
#     train_Y = train_Y.to(dtype=torch.float64)

#     return train_X, train_Y   


# #model initiation

# #here I will use a multi-output singletaskGP, since the objectives all come from the same training data and are independent (at least I assume it for now)

# train_X, train_Y = get_initial_dataset(parameter_bounds, num_points=7)

# print(train_X, train_Y)

# model = SingleTaskGP(train_X=train_X, train_Y=train_Y, input_transform=Normalize(d=4, bounds=parameter_bounds), outcome_transform=Standardize(m=2))

# mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model) 

# mll = fit_gpytorch_model(mll)

# posterior = model.posterior(train_X)

# acq_function = qLogNoisyExpectedHypervolumeImprovement(model=model, X_baseline=train_X, ref_point=torch.tensor([0.0, 0.0], dtype=torch.float64))

#         # here implementation of input constraints but not supported yet!
#         # 

# candidate, acq_value = optimize_acqf(
#     acq_function=acq_function,
#     bounds=parameter_bounds,
#     q=1,
#     num_restarts=40,
#     raw_samples=512,
#         )