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
from botorch.models import MultiTaskGP, SingleTaskGP, ModelList, ModelListGP


class MultiRGPEInitializer(BaseModel):
    
    def __init__(self, dataset: DataManager, weight_calculation_method: str):
        super().__init__(dataset)
        self.gp_model = None
        self.fitted_historic_gp_models = []
        self.target_gp_model = None
        self.n_mc_samples = None
        self.calculated_weights = None
        self.weight_calculation_method = weight_calculation_method


    def setup_model(self, n_mc_samples):

        if self.dataset_manager.initial_dataset.num_datapoints == 0:
            raise ValueError("At least one initial dataset must be loaded to initialize the model.")
        
        if not self.dataset_manager.historic_dataset_list:
            raise ValueError("No historic datasets found. Please provide at least one historic dataset for the Multi_RGPE!")
        
        if self.weight_calculation_method not in ["pareto_dominance", "objective_wise"]:
            raise ValueError('weight_calculation_method must be either "pareto_dominance" or "objective_wise".') 
        
        self.weight_calculation_method = self.weight_calculation_method

        print(f"Setting up Multi-RGPE model with {len(self.dataset_manager.historic_dataset_list)} historic models and one current model.")
        
        print(f"The weights are calculated using the {self.weight_calculation_method} method.")

        self.n_mc_samples = n_mc_samples

        self.initialize_and_train_historic_models()

        #initialize target GP modellist

        num_objectives = self.dataset_manager.output_dim

        modelllist_gp = []

        for objective in range(num_objectives):
            gp_model = SingleTaskGP(train_X=self.dataset_manager.initial_dataset.input_data,
                        train_Y=self.dataset_manager.initial_dataset.output_data[:, objective].unsqueeze(-1),
                        input_transform=Normalize(d=self.dataset_manager.input_dim), 
                        outcome_transform=Standardize(m=1))
            
            modelllist_gp.append(gp_model)

        target_modellist = ModelListGP(*modelllist_gp)

        # Fit the model
        mll_tar = SumMarginalLogLikelihood(target_modellist.likelihood, target_modellist)
        mll_tar = fit_gpytorch_mll(mll_tar)

        print(f"Target model initialized with {num_objectives} objectives.")

        self.target_gp_model = target_modellist

        if self.dataset_manager.initial_dataset.input_data.shape[0] < 3:
            num_objectives = self.dataset_manager.output_dim
            num_models = len(self.fitted_historic_gp_models) + 1  # Includes historic models and the current model

            # Create a tensor of shape (num_objectives, num_models) where each entry is 1 / num_models
            rank_weights = torch.ones(num_objectives, num_models, dtype=torch.float64) / num_models
            print(f"Since the number of initial data points is less than 3, all weights will be equal: {rank_weights}. Each weight corresponds to one historic model and one current model.")

        else:    

            if self.weight_calculation_method == "objective_wise":
                rank_weights = self.compute_rank_weights_objective_wise(train_X=self.dataset_manager.initial_dataset.input_data, 
                                                    train_Y=self.dataset_manager.initial_dataset.output_data, 
                                                    historic_model_list=self.fitted_historic_gp_models, 
                                                    n_mc_samples=self.n_mc_samples
                                                    )
            else:
                rank_weights = self.compute_rank_weights_pareto_dominance(train_X=self.dataset_manager.initial_dataset.input_data, 
                                                    train_Y=self.dataset_manager.initial_dataset.output_data, 
                                                    historic_model_list=self.fitted_historic_gp_models, 
                                                    n_mc_samples=self.n_mc_samples
                                                    )

            # Convert the rank weights to a tensor with dtype=torch.float64, otherwise error in qLogNoisyExpectedHypervolumeImprovement
            rank_weights = rank_weights.to(dtype=torch.float64)

            self.calculated_weights = rank_weights

            print(f"The calculated rank weights are: {rank_weights}. Each weight corresponds to one historic model and one current model per objective.")

        model_list = self.fitted_historic_gp_models + [self.target_gp_model]
        
        multi_rgpe_gp = self.setup_multiple_rgpe(model_list, rank_weights)

        self.gp_model = multi_rgpe_gp

        print(f"Model initialized. Number of objectives: {self.gp_model.num_outputs}.")
        

    def initialize_and_train_historic_models(self):

        for historic_dataset in self.dataset_manager.historic_dataset_list:

            historic_gp_model = self.initialize_single_gp_model(historic_dataset)
            
            mll = SumMarginalLogLikelihood(historic_gp_model.likelihood, historic_gp_model)

            mll = fit_gpytorch_mll(mll=mll)

            self.fitted_historic_gp_models.append(historic_gp_model)

        print(f"Setup of {len(self.fitted_historic_gp_models)} historic models completed. All are of type modellist and based on SingleTaskGP models.")
        

    def initialize_single_gp_model(self, historic_dataset):
        
        # Convert input_data and output_data to torch tensors with dtype=torch.float64
        train_X = torch.tensor(historic_dataset['input_data'], dtype=torch.float64)
        train_Y = torch.tensor(historic_dataset['output_data'], dtype=torch.float64)
        
        num_objectives = self.dataset_manager.output_dim

        modelllist_gp = []

        for objective in range(num_objectives):
            gp_model = SingleTaskGP(train_X=train_X,
                        train_Y=train_Y[:, objective].unsqueeze(-1),
                        input_transform=Normalize(d=self.dataset_manager.input_dim), 
                        outcome_transform=Standardize(m=1))
            
            modelllist_gp.append(gp_model)

        gp_modellist = ModelListGP(*modelllist_gp)

        return gp_modellist

    def compute_rank_weights_objective_wise(self, train_X, train_Y, historic_model_list, n_mc_samples):

        # Initialize an empty list to store ranking losses for all objectives and models
        ranking_losses_all_objectives = []

        # Loop over each historic model and compute the ranking loss for each objective
        for task in range(len(historic_model_list)):
            historic_model = historic_model_list[task]

            posterior = historic_model.posterior(train_X)

            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n_mc_samples]))
            base_f_samps = sampler(posterior).squeeze(-1).squeeze(-1)

            ranking_losses = []

            for objective in range(self.dataset_manager.output_dim):
                # Compute ranking loss for each objective
                ranking_loss = self.compute_ranking_loss_objective_wise(base_f_samps[:,:,objective], train_Y[:, objective])
                ranking_losses.append(ranking_loss)

            # Append the ranking losses for this model to the overall list
            ranking_losses_all_objectives.append(torch.stack(ranking_losses))

        # Also compute ranking losses for the target model
        target_f_samps = self.get_target_model_loocv_sample_preds(train_X=train_X, train_Y=train_Y, n_mc_samples=n_mc_samples)

        ranking_losses = []

        for objective in range(self.dataset_manager.output_dim):
            ranking_loss = self.compute_ranking_loss_objective_wise(target_f_samps[:,:,objective], train_Y[:, objective])
            ranking_losses.append(ranking_loss)

        # Append the ranking losses for the target model to the overall list
        ranking_losses_all_objectives.append(torch.stack(ranking_losses))

        # Convert the list of ranking losses into a tensor with shape (n_models, n_objectives, n_mc_samples)
        ranking_loss_tensor = torch.stack(ranking_losses_all_objectives)

        # Initialize rank_weights tensor of shape (n_objectives, n_models)
        rank_weights = torch.zeros(self.dataset_manager.output_dim, len(historic_model_list) + 1)

        # Determine the best model for each Monte Carlo sample (minimum ranking loss) per objective
        for objective in range(self.dataset_manager.output_dim):
            best_models = torch.argmin(ranking_loss_tensor[:, objective, :], dim=0)

            # Compute the proportion of samples for which each model is the best
            model_counts = torch.bincount(best_models, minlength=len(historic_model_list) + 1).float()
            rank_weights[objective, :] = model_counts / n_mc_samples

        #returned rank_weights is of shape ([d, n_models]) with d being the number of objectives and n_models being the number of models (historic and current model)
        return rank_weights
             
    
    def compute_rank_weights_pareto_dominance(self, train_X, train_Y, historic_model_list, n_mc_samples):

        ranking_losses = []

        # Compute ranking losses for each historic model
        for task in range(len(historic_model_list)):
            historic_model = historic_model_list[task]

            posterior = historic_model.posterior(train_X)

            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n_mc_samples]))
            
            base_f_samps = sampler(posterior).squeeze(-1).squeeze(-1)

            ranking_losses.append(self.compute_ranking_loss_pareto_dominance(base_f_samps, train_Y))

        target_f_samps = self.get_target_model_loocv_sample_preds(train_X=train_X, train_Y=train_Y, n_mc_samples=n_mc_samples)

        ranking_losses.append(self.compute_ranking_loss_pareto_dominance(target_f_samples=target_f_samps, true_train_Y=train_Y))

        # Convert the list of ranking losses into a tensor for easier processing
        ranking_loss_tensor = torch.stack(ranking_losses)

        # Determine the best model for each Monte Carlo sample (minimum ranking loss)
        best_models = torch.argmin(ranking_loss_tensor, dim=0)

        # Compute the proportion of samples for which each model is the best
        rank_weights = best_models.bincount(minlength=len(ranking_losses)).float() / n_mc_samples

        num_objectives = self.dataset_manager.output_dim

        # Stack the rank_weights to match the number of objectives
        rank_weights_stacked = torch.stack([rank_weights] * num_objectives, dim=0)

        return rank_weights_stacked


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
            loocv_modelllist_gp = []

            for objective in range(d):
                gp_model = SingleTaskGP(train_X=loocv_train_X,
                            train_Y=loocv_train_Y[:, objective].unsqueeze(-1),
                            input_transform=Normalize(d=self.dataset_manager.input_dim), 
                            outcome_transform=Standardize(m=1))
                
                loocv_modelllist_gp.append(gp_model)

            gp_modellist = ModelListGP(*loocv_modelllist_gp)

            # Fit the model
            mll = SumMarginalLogLikelihood(gp_modellist.likelihood, gp_modellist)
            mll = fit_gpytorch_mll(mll)

            # Predict the left-out point - Important: just for the one point here!
            posterior = gp_modellist.posterior(train_X[i:i+1])

            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n_mc_samples]))

            base_f_samps = sampler(posterior).squeeze(-2)

            # Store the prediction in the correct position
            loocv_preds[:, i, :] = base_f_samps

        return loocv_preds
    
    def compute_ranking_loss_objective_wise(self, target_f_samples, true_train_Y):

        #this function only calculates the ranking loss for one objective at a time and returns a tensor of shape ([n_mc_samples]) with n_mc_samples elements which is stacked afterwards

        #check that both the target_f_samples and true_train_Y have only one objective dimension
        if target_f_samples.shape[-1] !=  true_train_Y.shape[-1]:
            raise ValueError("The target_f_samples and true_train_Y must have the same number of points per objective.")
        
        # Extract the n and d dimensions from true_train_Y
        n_true_Y = true_train_Y.shape[0]

        ranking_losses = []

        # Iterate over all Monte Carlo samples
        n_mc_samples = target_f_samples.shape[0]

        for sample_idx in range(n_mc_samples):
            
            ranking_loss = 0

            for i in range(n_true_Y):
                for j in range(i + 1, n_true_Y):
                    # Get the predicted and true values for the pair (i, j)
                    pred_i, pred_j = target_f_samples[sample_idx, i], target_f_samples[sample_idx, j]
                    true_i, true_j = true_train_Y[i], true_train_Y[j]

                    # Compute the pairwise indicator function as per the equation
                    indicator = int((pred_i < pred_j) != (true_i < true_j))

                    # Accumulate the ranking loss
                    ranking_loss += indicator

            ranking_losses.append(ranking_loss)
        
        ranking_loss_tensor = torch.tensor(ranking_losses, dtype=torch.float64)

        return ranking_loss_tensor


    def compute_ranking_loss_pareto_dominance(self, target_f_samples, true_train_Y):

        #true_train_Y is a tensor of shape ([n,d]) with n * d elements

        # target_f_samples is a tensor of shape ([n_mc_samples, n, d]) with n_mc_samples * n * d elements  

        # Extract the n and d dimensions from true_train_Y
        n_true_Y, d_true_Y = true_train_Y.shape
        
        # Check that the second and third dimensions of target_f_samples match the shape of true_train_Y
        if target_f_samples.shape[1:] != (n_true_Y, d_true_Y):
            raise ValueError("The n and d dimensions of target_f_samples must match those of true_train_Y.")
        
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
    
    def setup_multiple_rgpe(self, model_list, rank_weights_stacked):

        multi_rgpe_modellist = []

        for objective in range(self.dataset_manager.output_dim):
            
            singletaskgp_list = []

            for model in model_list:
                singletaskgp_list.append(model.models[objective])

            # Extract the corresponding weights for the current objective
            objective_weights = rank_weights_stacked[objective, :]

            # Initialize RGPE for the current objective with the appropriate weights
            rgpe_singleobjective = RGPE(models=singletaskgp_list, weights=objective_weights)

            multi_rgpe_modellist.append(rgpe_singleobjective)

        #can be used with any number of single-output `GPyTorchModel`\s and the models can be of different types
        multi_rgpe_modellist = ModelListGP(*multi_rgpe_modellist)

        return multi_rgpe_modellist

    
    def train_initially_gp_model(self):

        if self.gp_model is None:
            raise ValueError("No GP model set. Please run an initiation first!")
        
        print(f"No training is done here, since the already trained models on the data were combined for a RGPE model")

        # mll = SumMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)

        # mll = fit_gpytorch_mll(mll=mll)

    def reinitialize_model(self, current_iteration: int):

        num_objectives = self.dataset_manager.output_dim

        modelllist_gp = []

        for objective in range(num_objectives):
            gp_model = SingleTaskGP(train_X=self.dataset_manager.initial_dataset.input_data,
                        train_Y=self.dataset_manager.initial_dataset.output_data[:, objective].unsqueeze(-1),
                        input_transform=Normalize(d=self.dataset_manager.input_dim), 
                        outcome_transform=Standardize(m=1))
            
            modelllist_gp.append(gp_model)

        target_modellist = ModelListGP(*modelllist_gp)

        # Fit the model
        mll_tar = SumMarginalLogLikelihood(target_modellist.likelihood, target_modellist)
        mll_tar = fit_gpytorch_mll(mll_tar)

        self.target_gp_model = target_modellist

        if self.dataset_manager.initial_dataset.input_data.shape[0] < 3:
            num_objectives = self.dataset_manager.output_dim
            num_models = len(self.fitted_historic_gp_models) + 1  # Includes historic models and the current model

            # Create a tensor of shape (num_objectives, num_models) where each entry is 1 / num_models
            rank_weights = torch.ones(num_objectives, num_models, dtype=torch.float64) / num_models
            print(f"Since the number of initial data points is less than 3, all weights will be equal: {rank_weights}. Each weight corresponds to one historic model and one current model.")

        else:
            if self.weight_calculation_method == "objective_wise":
                rank_weights = self.compute_rank_weights_objective_wise(train_X=self.dataset_manager.initial_dataset.input_data, 
                                                    train_Y=self.dataset_manager.initial_dataset.output_data, 
                                                    historic_model_list=self.fitted_historic_gp_models, 
                                                    n_mc_samples=self.n_mc_samples
                                                    )
            else:
                rank_weights = self.compute_rank_weights_pareto_dominance(train_X=self.dataset_manager.initial_dataset.input_data, 
                                                    train_Y=self.dataset_manager.initial_dataset.output_data, 
                                                    historic_model_list=self.fitted_historic_gp_models, 
                                                    n_mc_samples=self.n_mc_samples
                                                    )


            # Convert the rank weights to a tensor with dtype=torch.float64, otherwise error in qLogNoisyExpectedHypervolumeImprovement
            rank_weights = rank_weights.to(dtype=torch.float64)

            self.calculated_weights = rank_weights

            print(f"The calculated rank weights of the Multi-RGPE with {len(self.dataset_manager.historic_dataset_list)} historic models and one current model are: {rank_weights}. Each weight corresponds to one historic model and one current model.")

        model_list = self.fitted_historic_gp_models + [self.target_gp_model]
        
        multi_rgpe_gp = self.setup_multiple_rgpe(model_list, rank_weights)

        self.gp_model = multi_rgpe_gp

        print(f"Model reinitialized for iteration {current_iteration}. Number of objectives: {self.gp_model.num_outputs}. Number of points in the dataset: {self.dataset_manager.initial_dataset.num_datapoints}.") 


class RGPE(GP, GPyTorchModel):
    """
    Rank-weighted GP ensemble. Note: this class inherits from GPyTorchModel which provides an
        interface for GPyTorch models in botorch.
    """

    _num_outputs = 1

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
            posterior_mean = posterior.mean.squeeze(-1)
            posterior_cov = posterior.mvn.lazy_covariance_matrix
            # apply weight
            weight = non_zero_weights[non_zero_weight_idx]
            weighted_means.append(weight * posterior_mean)
            weighted_covars.append(posterior_cov * weight**2)
        # set mean and covariance to be the rank-weighted sum the means and covariances of the
        # base models and target model
        mean_x = torch.stack(weighted_means).sum(dim=0)
        covar_x = PsdSumLazyTensor(*weighted_covars)
        return MultivariateNormal(mean_x, covar_x)
    
