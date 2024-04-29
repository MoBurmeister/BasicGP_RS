import torch
from typing import Callable

# TODO: implement transform data function
# TODO: implement receive dataset from filepath function
# TODO: further testing, especially for multi input/output functions

class DatasetManager:
    def __init__(self, dtype: torch.dtype, filepath: str = None) -> None:
        self.filepath = filepath
        self.dtype = dtype
        self.dataset_func = None
        self.unscaled_data = None
        self.scaled_data = None
        self.scaling_dic = {}

    def func_create_dataset(self, dataset_func: Callable[..., torch.Tensor], num_datapoints: int, sampling_method: str = "grid", noise_level: float = 0.0, standardize: bool = True, min_max_scaling: bool = False, **kwargs):
        
        if sampling_method not in ['random', 'grid']:
            raise ValueError("Sampling method must be 'random' or 'grid'.")
        if standardize and min_max_scaling:
            raise ValueError("Specify only one type of scaling: standardize or min_max_scaling.")
        
        inputs = []
        if sampling_method == "random":
            for key, range_val in kwargs.items():
                inputs.append(torch.rand(num_datapoints) * (range_val[1] - range_val[0]) + range_val[0])
        elif sampling_method == "grid":
            grids = [torch.linspace(*range_val, steps=int(num_datapoints ** (1 / len(kwargs))))
                     for key, range_val in kwargs.items()]
            mesh = torch.meshgrid(*grids, indexing='ij')
            inputs = [item.flatten() for item in mesh]

        inputs = torch.stack(inputs, dim=1)

        if noise_level > 0:
            inputs = self.add_noise(inputs=inputs, noise_level=noise_level)

        inputs = inputs.to(self.dtype)

        outputs = dataset_func(inputs)
        outputs = outputs.to(self.dtype)

        self.unscaled_data = (inputs.clone(), outputs.clone())

        if standardize:
            inputs = self.data_scaling(inputs)
            outputs = self.data_scaling(outputs)
        if min_max_scaling:
            inputs = self.min_max_scaling(inputs)
            outputs = self.min_max_scaling(outputs)

        self.scaled_data = (inputs, outputs)

    def add_noise(self, inputs: torch.Tensor, noise_level: float):
        """ Adds Gaussian noise to input data.

        :param inputs: Input data tensor.
        :param noise_level: Standard deviation of Gaussian noise to add.
        :return: Inputs with noise added.
        """
        return inputs + noise_level * torch.randn_like(inputs)
    
    def data_scaling(self, inputs:torch.Tensor):
        mean = torch.mean(inputs, dim=0, keepdim=True)
        std = torch.std(inputs, dim=0, keepdim=True)
        self.scaling_dic['standard'] = {'mean': mean, 'std': std}
        return (inputs - mean) / std
    
    def min_max_scaling(self, inputs: torch.Tensor):
        """Scales the input tensor to the [0, 1] range."""
        min_val = torch.min(inputs, dim=0, keepdim=True).values
        max_val = torch.max(inputs, dim=0, keepdim=True).values
        self.scaling_dic['min_max'] = {'min': min_val, 'max': max_val}
        return (inputs - min_val) / (max_val - min_val)
    
    def transform_data(self):
        pass

    def train_test_split(self):
        pass