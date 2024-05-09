import torch
from typing import Callable

# TODO: implement transform data function
# TODO: implement receive dataset from filepath function
# TODO: Bounds integration, especially for the filepath acquiring of data
# TODO: further testing, especially for multi input/output functions
# TODO: what about more robust scaling methods? What about the scaling schedule?

# TODO: the **kwargs should be changed here into a range dict, which is a more clear way of initiating it

class DatasetManager:
    def __init__(self, dtype: torch.dtype, filepath: str = None) -> None:
        self.filepath = filepath
        self.dtype = dtype
        self.input_dim = None
        self.output_dim = None
        self.num_datapoints = None
        self.dataset_func = None
        self.unscaled_data = None
        self.bounds_list = []

    def func_create_dataset(self, dataset_func: Callable[..., torch.Tensor], num_datapoints: int, sampling_method: str = "grid", noise_level: float = 0.0, **kwargs):
        
        self.num_datapoints = num_datapoints

        if sampling_method not in ['random', 'grid']:
            raise ValueError("Sampling method must be 'random' or 'grid'.")
        
        inputs = []
        if sampling_method == "random":
            for key, range_val in kwargs.items():
                inputs.append(torch.rand(num_datapoints) * (range_val[1] - range_val[0]) + range_val[0])
            inputs = torch.stack(inputs, dim=1)
        elif sampling_method == "grid":
            for key, range_val in kwargs.items():
                inputs.append(torch.linspace(range_val[0], range_val[1], steps=num_datapoints))
            inputs = torch.stack(inputs, dim=1)

        self.setbounds(**kwargs)

        inputs = inputs.to(self.dtype)
        outputs, self.output_dim = dataset_func(inputs)

        if noise_level > 0:
            outputs = self.add_noise(outputs=outputs, noise_level=noise_level)

        outputs = outputs.to(self.dtype)

        self.unscaled_data = (inputs.clone(), outputs.clone())

        self.input_dim = sum(1 for key in kwargs if '_range' in key)

        self.check_shape(inputs, outputs)

        self.check_dimensions(inputs, outputs)

    def setbounds(self, **kwargs):
        self.bounds_list = [value for key, value in kwargs.items() if "range" in key]

    def add_noise(self, outputs: torch.Tensor, noise_level: float):
        """ Adds Gaussian noise to output data.

        :param outputs: Input data tensor.
        :param noise_level: Standard deviation of Gaussian noise to add.
        :return: outputs with noise added.
        """
        return outputs + noise_level * torch.randn_like(outputs)
    
    def check_shape(self, inputs: torch.Tensor, outputs: torch.Tensor):
        """
        Verifies that both inputs and outputs tensors have the shape ([n, d]) and
        that n is equivalent to self.num_datapoints.

        Args:
        inputs (torch.Tensor): The input tensor to be verified.
        outputs (torch.Tensor): The output tensor to be verified.

        Raises:
        ValueError: If the number of datapoints (n) in inputs or outputs does not match self.num_datapoints.
        ValueError: If inputs or outputs are not two-dimensional.
        """
        if inputs.ndim != 2:
            raise ValueError(f"Inputs should be 2-dimensional, got {inputs.ndim} dimensions")
        if outputs.ndim != 2:
            raise ValueError(f"Outputs should be 2-dimensional, got {outputs.ndim} dimensions")

        if inputs.shape[0] != self.num_datapoints:
            raise ValueError(f"Input tensor number of datapoints mismatch: expected {self.num_datapoints}, got {inputs.shape[0]}")
        if outputs.shape[0] != self.num_datapoints:
            raise ValueError(f"Output tensor number of datapoints mismatch: expected {self.num_datapoints}, got {outputs.shape[0]}")
      
    def check_dimensions(self, inputs: torch.Tensor, outputs: torch.Tensor):
        """
        Verifies that the inputs and outputs have dimensions that match the expected input_dim and output_dim.
        
        Args:
        inputs (torch.Tensor): The input tensor to be verified.
        outputs (torch.Tensor): The output tensor to be verified.
        
        Raises:
        ValueError: If dimensions do not match the expected dimensions.
        """
        if inputs.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {inputs.shape[1]}")
        if outputs.shape[1] != self.output_dim:
            raise ValueError(f"Output dimension mismatch: expected {self.output_dim}, got {outputs.shape[1]}")

    def train_test_split(self):
        pass