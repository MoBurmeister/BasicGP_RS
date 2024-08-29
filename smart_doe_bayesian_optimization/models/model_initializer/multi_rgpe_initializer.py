from models.gp_model import BaseModel
from data.create_dataset import DataManager
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
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import PsdSumLazyTensor
from gpytorch.likelihoods import LikelihoodList
from torch.nn import ModuleList


class MultiRGPEInitializer(BaseModel):
    
    def __init__(self, dataset: DataManager):
        super().__init__(dataset)
        self.gp_model = None
        self.fitted_historic_gp_models = []
        self.target_gp_model = None
        self.n_mc_samples = None


    def setup_model(self, n_mc_samples):

        if self.dataset_manager.initial_dataset.input_data.shape[0] < 3:
            raise ValueError("The number of initial datapoints must be at least 3 to initialize the model.")
        
        if len(self.dataset_manager.historic_dataset_list) == 0:
            raise ValueError("At least one historic dataset must be loaded to initialize the model.")
        
        self.n_mc_samples = n_mc_samples
        

        self.initialize_and_train_historic_models()

        #initialize target GP model

        target_gp_model = SingleTaskGP(
            train_X=self.dataset_manager.initial_dataset.input_data, 
            train_Y=self.dataset_manager.initial_dataset.output_data, 
            outcome_transform=Standardize(m=self.dataset_manager.initial_dataset.output_data.shape[1]), 
            input_transform=Normalize(d=self.dataset_manager.initial_dataset.input_data.shape[1])
        )

        self.target_gp_model = target_gp_model

        rank_weights = self.compute_rank_weights(train_X=self.dataset_manager.initial_dataset.input_data, 
                                                 train_Y=self.dataset_manager.initial_dataset.output_data, 
                                                 historic_model_list=self.fitted_historic_gp_models, 
                                                 n_mc_samples=n_mc_samples
                                                 )
        
      
        print(rank_weights)
        
        model_list = self.fitted_historic_gp_models + [target_gp_model]
        
        multi_rgpe_gp = RGPE(models=model_list, weights=rank_weights, num_outputs=self.dataset_manager.output_dim)

        self.gp_model = multi_rgpe_gp
        

    def initialize_and_train_historic_models(self):

        for historic_dataset in self.dataset_manager.historic_dataset_list:

            historic_gp_model = self.initialize_single_gp_model(historic_dataset)
            
            mll = ExactMarginalLogLikelihood(historic_gp_model.likelihood, historic_gp_model)

            mll = fit_gpytorch_mll(mll=mll)

            self.fitted_historic_gp_models.append(historic_gp_model)
        

    def initialize_single_gp_model(self, historic_dataset):
        
        # Convert input_data and output_data to torch tensors with dtype=torch.float64
        train_X = torch.tensor(historic_dataset['input_data'], dtype=torch.float64)
        train_Y = torch.tensor(historic_dataset['output_data'], dtype=torch.float64)
        
        # Initialize the model with the converted tensors
        gp_model = SingleTaskGP(
            train_X=train_X, 
            train_Y=train_Y, 
            outcome_transform=Standardize(m=train_Y.shape[1]), 
            input_transform=Normalize(d=train_X.shape[1])
        )
        return gp_model
    
    def compute_rank_weights(self, train_X, train_Y, historic_model_list, n_mc_samples):

        ranking_losses = []

        # Compute ranking losses for each historic model
        for task in range(len(historic_model_list)):
            historic_model = historic_model_list[task]

            posterior = historic_model.posterior(train_X)

            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n_mc_samples]))
            
            base_f_samps = sampler(posterior).squeeze(-1).squeeze(-1)

            ranking_losses.append(self.compute_ranking_loss(base_f_samps, train_Y))

        target_f_samps = self.get_target_model_loocv_sample_preds(train_X=train_X, train_Y=train_Y, n_mc_samples=n_mc_samples)

        ranking_losses.append(self.compute_ranking_loss(target_f_samples=target_f_samps, true_train_Y=train_Y))

        print(ranking_losses)

        # Convert the list of ranking losses into a tensor for easier processing
        ranking_loss_tensor = torch.stack(ranking_losses)

        # Determine the best model for each Monte Carlo sample (minimum ranking loss)
        best_models = torch.argmin(ranking_loss_tensor, dim=0)

        # Compute the proportion of samples for which each model is the best
        rank_weights = best_models.bincount(minlength=len(ranking_losses)).float() / n_mc_samples

        return rank_weights


    def get_target_model_loocv_sample_preds(self, train_X, train_Y, n_mc_samples):

        n = train_X.shape[0]  # Number of data points
        d = train_Y.shape[1]  # Dimensionality of the output

        # Initialize an empty tensor to store the predictions
        loocv_preds = torch.empty(n_mc_samples, n, d, device=train_X.device)

        # Iterate over each data point to perform LOOCV
        for i in range(n):
            # Create new training sets excluding the i-th point
            loocv_train_X = torch.cat([train_X[:i], train_X[i+1:]])
            loocv_train_Y = torch.cat([train_Y[:i], train_Y[i+1:]])

            # Initialize a new GP model for this subset of data
            loocv_gp_model = SingleTaskGP(
                train_X=loocv_train_X,
                train_Y=loocv_train_Y,
                outcome_transform=Standardize(m=loocv_train_Y.shape[1]),
                input_transform=Normalize(d=loocv_train_X.shape[1])
            )

            # Fit the model (e.g., using fit_gpytorch_mll)
            mll = ExactMarginalLogLikelihood(loocv_gp_model.likelihood, loocv_gp_model)
            fit_gpytorch_mll(mll)

            # Predict the left-out point - Important: just for the one point here!
            posterior = loocv_gp_model.posterior(train_X[i:i+1])

            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n_mc_samples]))

            base_f_samps = sampler(posterior).squeeze(-2)

            # Store the prediction in the correct position
            loocv_preds[:, i, :] = base_f_samps

        return loocv_preds

    def compute_ranking_loss(self, target_f_samples, true_train_Y):

        #true_train_Y is a tensor of shape ([n,d]) with n * d elements

        # target_f_samples is a tensor of shape ([n_mc_samples, n, d]) with n_mc_samples * n * d elements  

        # Extract the n and d dimensions from true_train_Y
        n_true_Y, d_true_Y = true_train_Y.shape
        
        # Check that the second and third dimensions of target_f_samples match the shape of true_train_Y
        if target_f_samples.shape[1:] != (n_true_Y, d_true_Y):
            raise ValueError("The n and d dimensions of target_f_samples must match those of true_train_Y.")

        print(f"True train Y: {true_train_Y.tolist()}")
        print(f"Target f samples: {target_f_samples.tolist()}")
        
        # Initialize a list to store ranking losses for each sample
        ranking_losses = []

        # Iterate over all Monte Carlo samples
        n_mc_samples = target_f_samples.shape[0]

          # Initialize a counter for the total number of evaluations

        for sample_idx in range(n_mc_samples):
            ranking_loss = 0
            # Iterate over all pairs of training examples

            total_evaluations = 0

            for i in range(n_true_Y):
                for j in range(i + 1, n_true_Y):
                    # Get the predicted and true values for the pair (i, j)
                    pred_i, pred_j = target_f_samples[sample_idx, i], target_f_samples[sample_idx, j]
                    true_i, true_j = true_train_Y[i], true_train_Y[j]

                    # Compute Pareto dominance for both predicted and true values
                    pred_dominance = self.get_pareto_dominance(pred_i, pred_j)
                    true_dominance = self.get_pareto_dominance(true_i, true_j)

                    # Increment ranking loss if the predicted dominance doesn't match the true dominance
                    if pred_dominance != true_dominance:
                        ranking_loss += 1
                    
                    total_evaluations += 1
            
            #print(f"Total number of evaluations: {total_evaluations}")

            # Append the ranking loss for this Monte Carlo sample to the list
            ranking_losses.append(ranking_loss)

        

        ranking_loss_tensor = torch.tensor(ranking_losses, dtype=torch.float64)

        print(f"Ranking losses: {ranking_loss_tensor}")
    
        return ranking_loss_tensor


    def get_pareto_dominance(self, point_1, point_2):

        # Check if point_1 dominates point_2
        dominates = (point_1 <= point_2).all() and (point_1 < point_2).any()
        
        if dominates:
            return 1  # point_1 dominates point_2
        
        # Check if point_2 dominates point_1
        dominated = (point_2 <= point_1).all() and (point_2 < point_1).any()
        
        if dominated:
            return -1  # point_2 dominates point_1
        
        return 0  # Neither point dominates the other


    
    def train_initially_gp_model(self):

        if self.gp_model is None:
            raise ValueError("No GP model set. Please run an initiation first!")
        
        print(f"No training is done here, since the already trained models on the data were combined for a RGPE model")

        # mll = SumMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)

        # mll = fit_gpytorch_mll(mll=mll)




class RGPE(GP, GPyTorchModel):
    """
    Rank-weighted GP ensemble. Note: this class inherits from GPyTorchModel which provides an
        interface for GPyTorch models in botorch.
    """

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
        weighted_means = []
        weighted_covariances = []
        
        for i, model in enumerate(self.models):
            posterior = model(x)
            mean_x = posterior.mean
            covar_x = posterior.covariance_matrix
            
            # Compute weighted mean and covariance
            weighted_means.append(self.weights[i] * mean_x)
            weighted_covariances.append(self.weights[i]**2 * covar_x)
        
        # Sum the weighted means and covariances
        mean_x = sum(weighted_means)
        covar_x = sum(weighted_covariances)
        return MultivariateNormal(mean_x, covar_x)