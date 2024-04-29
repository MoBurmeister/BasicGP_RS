from typing import Union
from botorch.models import SingleTaskGP, HeteroskedasticSingleTaskGP, FixedNoiseGP
from torch.optim import Adam, SGD
from gpytorch.mlls import ExactMarginalLogLikelihood, MarginalLogLikelihood
import torch
from gpytorch.models import ExactGP


# TODO: do I need different kind of training functions for the model? Especially for single/multi input etc.
# TODO: what about batching and batch sizes?
# TODO: adjust training messages -> to get the lengthscale is not possible 
# TODO: what about maximization/minimization? Relevant here? -> probably relevant in the mll?
# TODO: implement stopping criteria: Either when reaching no significant loss improvement or whne lengthscale/noise is converging -> Gaussian processes not so prone to overfitting

def training_gp_model(gp_model: ExactGP, 
                      optimizer: Union[Adam, SGD], 
                      mll: Union[ExactMarginalLogLikelihood, MarginalLogLikelihood], 
                      train_X: torch.Tensor, 
                      train_Y: torch.Tensor, 
                      num_epochs: int) -> Union[SingleTaskGP, HeteroskedasticSingleTaskGP, FixedNoiseGP]:
    """
    Trains a Gaussian Process (GP) model using gradient descent optimization to minimize the 
    negative marginal log likelihood (MLL).

    This function iteratively adjusts the model's parameters to improve its prediction accuracy
    on the provided training data. Training involves a forward pass to compute the GP's predictions,
    calculating the loss as the negative MLL, and using backpropagation to update the model's parameters.

    Parameters:
    - gp_model (Union[SingleTaskGP, HeteroskedasticSingleTaskGP, FixedNoiseGP]): The Gaussian Process model to train.
      This could be a standard GP, a heteroskedastic GP, or a GP with fixed noise levels.
    - optimizer (Union[Adam, SGD]): The optimization algorithm to use for training. Could be either Adam or SGD.
    - mll (Union[ExactMarginalLogLikelihood, MarginalLogLikelihood]): The Marginal Log Likelihood object used 
      for computing the training loss. Exact or approximate depending on the GP model used.
    - train_X (torch.Tensor): The training inputs (features), typically a tensor of shape (n_samples, n_features).
    - train_Y (torch.Tensor): The training outputs (targets), typically a tensor of shape (n_samples,).
    - num_epochs (int): The number of training iterations to perform.

    Returns:
    - Union[SingleTaskGP, HeteroskedasticSingleTaskGP, FixedNoiseGP]: The trained GP model.
    """
    gp_model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = gp_model(train_X)
        loss = -mll(output, gp_model.train_targets)
        loss.backward()
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:>3}/{num_epochs} - Loss: {loss.item():>4.3f} "
                #f"lengthscale: {gp_model.covar_module.base_kernel.lengthscale.item():>4.3f} "
                f"noise: {gp_model.likelihood.noise.item():>4.3f}"
            )
        optimizer.step()

    return gp_model
