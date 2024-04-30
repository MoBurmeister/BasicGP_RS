import torch
from typing import Callable

# TODO: implement transform data function
# TODO: implement receive dataset from filepath function
# TODO: further testing, especially for multi input/output functions

class DatasetManager:
    def __init__(self, dtype: torch.dtype, filepath: str = None) -> None:
        self.filepath = filepath
        self.dtype = dtype
        self.input_dim = None
        self.output_dim = None
        self.num_datapoints = None
        self.dataset_func = None
        self.unscaled_data = None
        self.scaled_data = None
        self.scaling_dict = {}

    def func_create_dataset(self, dataset_func: Callable[..., torch.Tensor], num_datapoints: int, sampling_method: str = "grid", noise_level: float = 0.0, standardize: bool = True, min_max_scaling: bool = False, **kwargs):
        
        self.num_datapoints = num_datapoints

        if sampling_method not in ['random', 'grid']:
            raise ValueError("Sampling method must be 'random' or 'grid'.")
        if standardize and min_max_scaling:
            raise ValueError("Specify only one type of scaling: standardize or min_max_scaling.")
        
        inputs = []
        if sampling_method == "random":
            for key, range_val in kwargs.items():
                inputs.append(torch.rand(num_datapoints) * (range_val[1] - range_val[0]) + range_val[0])
            inputs = torch.stack(inputs, dim=1)
        elif sampling_method == "grid":
            for key, range_val in kwargs.items():
                inputs.append(torch.linspace(range_val[0], range_val[1], steps=num_datapoints))
            inputs = torch.stack(inputs, dim=1)

        inputs = inputs.to(self.dtype)
        outputs, self.output_dim = dataset_func(inputs)

        if noise_level > 0:
            outputs = self.add_noise(outputs=outputs, noise_level=noise_level)

        outputs = outputs.to(self.dtype)

        self.unscaled_data = (inputs.clone(), outputs.clone())

        if standardize:
            inputs, outputs = self.data_scaling(inputs, outputs)
        if min_max_scaling:
            print("not implemented yet")

        self.input_dim = sum(1 for key in kwargs if '_range' in key)

        self.check_shape(inputs, outputs)

        self.check_dimensions(inputs, outputs)

        self.scaled_data = (inputs, outputs)

    def add_noise(self, outputs: torch.Tensor, noise_level: float):
        """ Adds Gaussian noise to output data.

        :param outputs: Input data tensor.
        :param noise_level: Standard deviation of Gaussian noise to add.
        :return: outputs with noise added.
        """
        return outputs + noise_level * torch.randn_like(outputs)
    
    def data_scaling(self, inputs: torch.Tensor, outputs: torch.Tensor):
        mean_inputs = torch.mean(inputs, dim=0, keepdim=True)
        std_inputs = torch.std(inputs, dim=0, keepdim=True)
        scaled_inputs = (inputs - mean_inputs) / std_inputs

        mean_outputs = torch.mean(outputs, dim=0, keepdim=True)
        std_outputs = torch.std(outputs, dim=0, keepdim=True)
        scaled_outputs = (outputs - mean_outputs) / std_outputs

        self.scaling_dict = {
            'inputs': {'method': 'standardize', 'mean': mean_inputs, 'std': std_inputs},
            'outputs': {'method': 'standardize', 'mean': mean_outputs, 'std': std_outputs}
        }

        return scaled_inputs, scaled_outputs
    
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
    
    
    def min_max_scaling(self, inputs: torch.Tensor):
        print("min_max_scaling not implemented yet")
        pass
    
    def transform_data(self):
        pass

    def train_test_split(self):
        pass