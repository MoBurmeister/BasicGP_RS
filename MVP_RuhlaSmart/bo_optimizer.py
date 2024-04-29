#necessary imports
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, LogExpectedImprovement
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

#General setup:
#use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
print(device)

#training loop: trains the model on the current available data
def train_loop(gp_model: SingleTaskGP, mll: ExactMarginalLogLikelihood, optimizer: Adam, train_X: torch.Tensor, train_Y: torch.Tensor, num_epochs: int):
    gp_model.train()

    for epoch in range(num_epochs):
        #clear gradients
        optimizer.zero_grad()
        #forward pass through the model to obtain the output MultivariateNormal
        output = gp_model(train_X)
        #Compute negative marginal log likelihood
        loss = -mll(output, gp_model.train_targets)
        #back prop gradients
        loss.backward()
        #print every 10 iterations
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:>3}/{num_epochs} - Loss: {loss.item():>4.3f} "
                f"lengthscale: {gp_model.covar_module.base_kernel.lengthscale.item():>4.3f} "
                f"noise: {gp_model.likelihood.noise.item():>4.3f}"
            )
        optimizer.step()
    
    return gp_model

#define acquisition function and optimize it based on the currently trained model: returns next parameter setting to observe in real world

#define acq_function:

def general_acq_function(gp_model: SingleTaskGP, train_Y: torch.tensor):
    #EI = ExpectedImprovement(model=gp_model, best_f=train_Y.max(), maximize=True)
    LEI = LogExpectedImprovement(model=gp_model, best_f=train_Y.max(), maximize=True)
    return LEI

#add bounds as input?

def return_next_parameter_setting(gp_model: SingleTaskGP, train_X: torch.Tensor, train_Y: torch.Tensor, bounds: torch.tensor):
    #acquisition function set here to maximization!
    
    EI = general_acq_function(gp_model, train_Y)
    candidate, acq_value = optimize_acqf(
        acq_function=EI,
        bounds=bounds,
        #looking for the next single best point:
        q=1,
        #number of times the optimization restarts to avoid local maxima:
        num_restarts=80,
        #number of randomly sampled points used to initilize the optimization of the acquisition function
        raw_samples=400,  # Vary depending on the problem
    )

    print(f"Next suggested x-point: {candidate}")
    return candidate

#acquire new observation data

def acquire_datapoint(candidate: torch.Tensor, function: callable):
    #extract scalar value
    point_to_evaluate = candidate.squeeze()
    #extract point - add dimension to be able to use it in the gt_function later
    function_value = function(point_to_evaluate.unsqueeze(-1))
    function_value = function_value.unsqueeze(1)
    print(f"Next acquired y-point: {function_value}")
    return function_value

#function to show the currently trained model and the acquisition function graph below it in a seperate 

def plot_gp_and_acquisition(model, train_X, train_Y, bounds):
    # Set model to eval mode
    model.eval()

    # Define points for plotting
    with torch.no_grad():
        # Range of points for plotting
        test_x = torch.linspace(bounds[0, 0], bounds[1, 0], 100).unsqueeze(-1)
        test_x = test_x.to(train_X.device, dtype=train_X.dtype)
        test_x = test_x.unsqueeze(-2)  # Add batch dimension for acquisition function
        posterior = model.posterior(test_x)
        mean = posterior.mean.squeeze(-1)
        # Get lower and upper confidence bounds (2 standard deviations)
        lower, upper = posterior.mvn.confidence_region()

    # Create figure for Gaussian Process plot
    fig_gp, ax_gp = plt.subplots(figsize=(10, 4))
    test_x_plot = test_x.squeeze(-2).detach()  # Adjust shape for plotting
    ax_gp.plot(train_X.cpu().numpy(), train_Y.cpu().numpy(), 'ko')
    ax_gp.plot(test_x_plot.cpu().numpy(), mean.cpu().numpy(), 'b')
    ax_gp.fill_between(test_x_plot.cpu().numpy().flatten(), lower.cpu().numpy().flatten(), upper.cpu().numpy().flatten(), alpha=0.5)
    ax_gp.set_title("Gaussian Process and Observed Data")
    ax_gp.legend(['Observed Data', 'Mean', 'Confidence'])

    EI = general_acq_function(model, train_Y)

    # Create figure for Acquisition Function plot
    fig_acq, ax_acq = plt.subplots(figsize=(10, 4))
    acq_values = EI(test_x).detach()
    ax_acq.plot(test_x_plot.cpu().numpy(), acq_values.cpu().numpy(), 'g-')
    ax_acq.set_title("Acquisition Function")
    ax_acq.set_xlabel("Input space")
    ax_acq.set_ylabel("Acquisition Value")

    return fig_gp, fig_acq

#Update old dataset: 

def update_dataset(old_data_X: torch.tensor, old_data_Y: torch.tensor, new_single_X: torch.tensor, new_single_Y: torch.tensor):
    
    new_train_X = torch.cat([old_data_X, new_single_X], 0)
    new_train_Y = torch.cat([old_data_Y, new_single_Y], 0)

    return new_train_X, new_train_Y

#function to implement the new point in the training data and refine the model: condition_on_observations

def refine_gp_model(gp_model: SingleTaskGP, new_train_X, new_train_Y):
    #return the refined model:
    # TODO: detaching the data is necessary?
    return gp_model.condition_on_observations(new_train_X, new_train_Y)

#final function to perform the next iteration of the BO with GP
def perform_next_iteration(gt_function: callable, gp_model: SingleTaskGP, train_X: torch.tensor, train_Y: torch.tensor, bounds: torch.tensor):
    x_dataset = train_X
    y_dataset = train_Y
    #return the next parameter setup to evaluate from the acquisition function
    new_x_value = return_next_parameter_setting(gp_model, x_dataset, y_dataset, bounds)
    #return the corresponding y-value for the x-parameter-setup
    new_y_value = acquire_datapoint(new_x_value, gt_function)
    #update the dataset
    new_train_X, new_train_Y = update_dataset(x_dataset, y_dataset, new_x_value, new_y_value)
    #refine the new model  
    updated_model = refine_gp_model(gp_model, new_train_X, new_train_Y)
    #receive both plots:
    model_plot, acq_plot = plot_gp_and_acquisition(updated_model, new_train_X, new_train_Y, bounds)

    return model_plot, acq_plot, new_x_value, new_y_value, new_train_X, new_train_Y, updated_model