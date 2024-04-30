from gpytorch.models import ExactGP
import torch

# TODO IMPORTANT: Consider the scaling of the data!
# TODO IMPORTANT: Consider the different dimensions in the data!
# TODO add support for mulit output (y1, y2, ...)

class GPVisualizer:

    @staticmethod
    def visualize_model_pdp(gp_model: ExactGP, train_X: torch.tensor, train_Y: torch.tensor):
        #pdp for partial dependence plots -> show one variable while averaging the others
        pass

