from gpytorch import kernels
from gpytorch.priors import GammaPrior

# TODO: what about combined kernels or stacked kernels?
# TODO: potentially add parameter checking for the kernel parameters, if they are in the kernel


class KernelFactory:

    @staticmethod
    def create_kernel(kernel_dict: dict) -> kernels.Kernel:

        kernel_type = kernel_dict["kernel_type"]
        kernel_params = parse_kernel_params(kernel_dict["kernel_params"])

        if kernel_type == 'RBF':
            return kernels.RBFKernel(**kernel_params)
        elif kernel_type == 'Matern':
            return kernels.MaternKernel(**kernel_params)
        elif kernel_type == 'Periodic':
            return kernels.PeriodicKernel(**kernel_params)
        elif kernel_type == 'Linear':
            return kernels.LinearKernel(**kernel_params)
        elif kernel_type == 'Polynomial':
            return kernels.PolynomialKernel(**kernel_params)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        

def parse_kernel_params(kernel_params):
    parsed_params = {}
    for key, value in kernel_params.items():
        if key == 'lengthscale_prior':
            prior_type = value['lengthscale_prior_type']
            # TODO: add other necessary prior types or variable types for handover
            if prior_type == 'GammaPrior':
                parsed_params['lengthscale_prior'] = GammaPrior(value['min_val'], value['max_val'])
        else:
            parsed_params[key] = value
    return parsed_params