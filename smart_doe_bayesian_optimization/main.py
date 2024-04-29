import models
from models.gp_model import BaseGPModel
from models.kernel_factory import KernelFactory
from gpytorch.priors import NormalPrior
from models.likelihood_factory import LikelihoodFactory
from gpytorch.constraints import GreaterThan


#Create a prior for the lengthscale
lengthscale_prior = NormalPrior(loc=1.0, scale=0.1)

# Instantiate an RBF Kernel with a lengthscale prior and ARD for 3 dimensions
rbf_kernel = KernelFactory.create_kernel(
    'RBF', 
    ard_num_dims=3, 
    lengthscale_prior=lengthscale_prior
)
# print(rbf_kernel.ard_num_dims)

gp_likelihood = LikelihoodFactory.create_likelihood(
    'Gaussian',
    noise_constraint = GreaterThan(1e-4)
)

print(gp_likelihood)

#first_gp = BaseGPModel("SingleTaskGP", "ExactMarginalLogLikelihood", "adam", )