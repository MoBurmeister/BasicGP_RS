import torch
from typing import Callable

# TODO: implement transform data function
# TODO: implement receive dataset from filepath function
# TODO: Bounds integration, especially for the filepath acquiring of data
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
        self.bounds_list = []

    def func_create_dataset(self, dataset_func: Callable[..., torch.Tensor], num_datapoints: int, sampling_method: str = "grid", noise_level: float = 0.0, scaling_input: str ='normalize', scaling_output: str = 'standardize', **kwargs):
        
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

        inputs = self.apply_scaling(inputs, scaling_input, 'inputs')
        outputs = self.apply_scaling(outputs, scaling_output, 'outputs')

        self.input_dim = sum(1 for key in kwargs if '_range' in key)

        self.check_shape(inputs, outputs)

        self.check_dimensions(inputs, outputs)

        self.scaled_data = (inputs, outputs)

    def setbounds(self, **kwargs):
        self.bounds_list = [value for key, value in kwargs.items() if "range" in key]

    def add_noise(self, outputs: torch.Tensor, noise_level: float):
        """ Adds Gaussian noise to output data.

        :param outputs: Input data tensor.
        :param noise_level: Standard deviation of Gaussian noise to add.
        :return: outputs with noise added.
        """
        return outputs + noise_level * torch.randn_like(outputs)
    
    def apply_scaling(self, data: torch.Tensor, method: str, data_name: str):

        if method == 'normalize':
            return self.data_normalize(data, data_name)
        elif method == 'standardize':
            return self.standardize_data(data, data_name)
        elif method == 'default':
            min_values = torch.min(data, dim=0, keepdim=True).values
            max_values = torch.max(data, dim=0, keepdim=True).values
            self.scaling_dict[data_name] = {
                'method': 'default',
                'params': None,
                'scaled_bounds': torch.stack((min_values, max_values), dim=0),
                'original_bounds': torch.stack((min_values, max_values), dim=0) 
            }
            return data
        else:
            raise ValueError(f"{method} not implemented as a scaling function for {data_name}")
    
    def standardize_data(self, inputs: torch.Tensor, data_name: str):
        mean_inputs = torch.mean(inputs, dim=0)
        std_inputs = torch.std(inputs, dim=0)
        scaled_inputs = (inputs - mean_inputs) / std_inputs

        if data_name == 'inputs':
            original_bounds = torch.tensor(self.bounds_list).t()
            scaled_lower_bounds = (original_bounds[0] - mean_inputs) / std_inputs
            scaled_upper_bounds = (original_bounds[1] - mean_inputs) / std_inputs
            scaled_bounds = torch.stack((scaled_lower_bounds, scaled_upper_bounds), dim=0)
        else:
            scaled_bounds = None
            original_bounds = None


        self.scaling_dict[data_name] = {
            'method': 'standardize',
            'params': {'mean': mean_inputs, 'std': std_inputs},
            'scaled_bounds': scaled_bounds,
            'original_bounds': original_bounds
        }

        return scaled_inputs
    
    #the min and max need to be the predefined range beginnings/ends!

    def data_normalize(self, inputs: torch.Tensor, data_name: str):
        min_inputs = torch.tensor([bound[0] for bound in self.bounds_list])
        max_inputs = torch.tensor([bound[1] for bound in self.bounds_list])
        normalized_inputs = (inputs - min_inputs) / (max_inputs - min_inputs)

        if data_name == 'inputs':
            original_bounds = torch.tensor(self.bounds_list).t()
            scaled_lower_bounds = torch.zeros(inputs.shape[1])
            scaled_upper_bounds = torch.ones(inputs.shape[1])
            scaled_bounds = torch.stack((scaled_lower_bounds, scaled_upper_bounds), dim=0)
        else:
            scaled_bounds = None
            original_bounds = None

        self.scaling_dict[data_name] = {
            'method': 'normalize',
            'params': {'min': min_inputs, 'max': max_inputs},
            'scaled_bounds': scaled_bounds,
            'original_bounds': original_bounds
        }

        return normalized_inputs
    
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
    
    def transform_data(self):
        pass

    def train_test_split(self):
        pass