from gpytorch import kernels

# TODO: what about combined kernels or stacked kernels?

class KernelFactory:

    @staticmethod
    def create_kernel(kernel_type: str, **kwargs):
        """
        Creates a kernel based on the specified type and parameters.
        
        Args:
            kernel_type (str): The type of kernel to create. Examples include 'RBF', 'Matern', 'Periodic', etc.
            **kwargs: Arbitrary keyword arguments, mainly for kernel configuration such as lengthscale.
        
        Returns:
            gpytorch.kernels.Kernel: The instantiated kernel object.
        
        Raises:
            ValueError: If the kernel type is unknown or parameters are missing.
        """
        if kernel_type == 'RBF':
            return kernels.RBFKernel(**kwargs)
        elif kernel_type == 'Matern':
            return kernels.MaternKernel(**kwargs)
        elif kernel_type == 'Periodic':
            return kernels.PeriodicKernel(**kwargs)
        elif kernel_type == 'Linear':
            return kernels.LinearKernel(**kwargs)
        elif kernel_type == 'Polynomial':
            return kernels.PolynomialKernel(**kwargs)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")