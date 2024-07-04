import torch
from typing import Callable, List, Tuple
from utils.checking_utils import check_type
from data.dataset import Dataset
import pickle

class DataManager:

    def __init__(self, dtype: torch.dtype = torch.float64):
        self.historic_data_loader = HistoricDataLoader(dtype=dtype)
        self.initial_data_loader = InitialDataLoader(dtype=dtype)
        self.input_dim = None
        self.output_dim = None
        self.initial_dataset = None
        #init continues below:
        '''
        How the dataset looks like: 

        main_dict = {
        'dict1': {
            'identifier': 1,
            'input_dataset': torch.randn(10, 5),  # Example tensor of shape [n, d]
            'output_dataset': torch.randn(10, 5)  # Example tensor of shape [n, d]
            'bounds': torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]])  # Example tensor of shape [2, d]
        },
        '''
        self.historic_datasets = []
        self.dtype = dtype
    
    def load_initial_dataset(self, dataset_func: Callable[..., torch.Tensor], num_datapoints:int, bounds: List[tuple], sampling_method: str = "grid", noise_level: float = 0.0):
        initial_dataset = self.initial_data_loader.load_dataset(dataset_func, num_datapoints, bounds, sampling_method, noise_level)
        check_type(initial_dataset, Dataset)
        self.initial_dataset = initial_dataset
        self.set_check_input_output_dim(initial_dataset.input_dim, initial_dataset.output_dim)
        print("Initial dataset added (and old one discarded):")
        print(f"Number of datapoints: {self.initial_dataset.input_data.shape[0]}")
        print(f"Number of input dimensions: {self.initial_dataset.input_data.shape[1]}")
        print(f"Number of output dimensions: {self.initial_dataset.output_data.shape[1]}")
        print('-' * 50)

    def load_historic_dataset(self, dataset_path: str):
        # each historic dataset itself carries a one model per objective? The model is then defined by cov module and mean? Or more the cov matrix
        # the historic datasets should rather probably just carry the data points, bounds etc. Initializ model here? 

        # Load the historic dataset from the pickle file
        with open(dataset_path, 'rb') as f:
            historic_dict_list = pickle.load(f)

        for hist_dataset in historic_dict_list:
            transformed_dataset = self.historic_data_loader.load_dataset(hist_dataset)
            self.set_check_input_output_dim(transformed_dataset.input_dim, transformed_dataset.output_dim)
            self.historic_datasets.append(transformed_dataset)
            print(f"Historic dataset added:")
            print(f"Identifier: {transformed_dataset.identifier}")
            print(f"Number of datapoints: {transformed_dataset.input_data.shape[0]}")
            print(f"Number of input dimensions: {transformed_dataset.input_data.shape[1]}")
            print(f"Number of output dimensions: {transformed_dataset.output_data.shape[1]}")
            print('-' * 50)


    def set_check_input_output_dim(self, input_dim: int, output_dim: int):
        
        if self.input_dim is not None and self.output_dim is not None:
            if self.input_dim != input_dim or self.output_dim != output_dim:
                raise ValueError("Input and output dimensions must match the existing dimensions.")
        else:
            self.input_dim = input_dim
            self.output_dim = output_dim
    
class InitialDataLoader:

    #here without classmethod, since there can be the case that I need costum configs for these intiated loaders 

    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype

    def load_dataset(self, dataset_func: Callable[..., torch.Tensor], num_datapoints:int, bounds: List[tuple], sampling_method: str = "grid", noise_level: float = 0.0) -> Dataset:
        """
        Load a dataset using the specified dataset function and parameters.
        Important: bounds are provided as a touple here, need to be converted into correct ([2, d]) shape via the convert_bounds_to_tensor function.

        Args:
            dataset_func (Callable[..., torch.Tensor]): A function that generates the dataset.
            num_datapoints (int): The number of datapoints to generate.
            bounds (List[tuple]): The bounds for each input dimension.
            sampling_method (str, optional): The method for sampling inputs. Defaults to "grid".
            noise_level (float, optional): The level of noise to add to the outputs. Defaults to 0.0.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            ValueError: If the sampling method is not 'random' or 'grid'.
        """
        
        num_datapoints = num_datapoints

        if sampling_method not in ['random', 'grid']:
            raise ValueError("Sampling method must be 'random' or 'grid'.")

        inputs = []
        if sampling_method == "random":
            for touple in bounds:
                inputs.append(torch.rand(num_datapoints) * (touple[1] - touple[0]) + touple[0])
            inputs = torch.stack(inputs, dim=1)
        elif sampling_method == "grid":
            for touple in bounds:
                inputs.append(torch.linspace(touple[0], touple[1], steps=num_datapoints))
            inputs = torch.stack(inputs, dim=1)

        bounds_tensor = self.convert_bounds_to_tensor(bounds)

        # TODO: do I still later need this _?

        output, _ = dataset_func(inputs)

        if noise_level > 0:
            outputs = self.add_noise(outputs=outputs, noise_level=noise_level)

        initial_datset = Dataset(input_data=inputs, output_data=output, bounds=bounds_tensor, datamanager_type="initial")

        return initial_datset


    def convert_bounds_to_tensor(self, bounds: List[Tuple[float, float]]) -> torch.Tensor:
        """
        Convert a list of bounds (min, max) into a tensor with shape [2, d].
        
        Args:
            bounds (List[Tuple[float, float]]): A list of tuples, where each tuple contains two floats 
                                                representing the minimum and maximum bounds.
        
        Returns:
            torch.Tensor: A tensor of shape [2, d], where d is the number of bounds. The first row 
                        contains the minimum values and the second row contains the maximum values.
                        
        Example:
            >>> bounds = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
            >>> convert_bounds_to_tensor(bounds)
            tensor([[1.0, 3.0, 5.0],
                    [2.0, 4.0, 6.0]])
        """
        # Separate the min and max values
        min_values = [b[0] for b in bounds]
        max_values = [b[1] for b in bounds]
        
        # Create a tensor from the lists of min and max values
        tensor = torch.tensor([min_values, max_values], dtype=self.dtype)
        
        return tensor
    
    def add_noise(self, outputs: torch.Tensor, noise_level: float):
        """ Adds Gaussian noise to output data.

        :param outputs: Input data tensor.
        :param noise_level: Standard deviation of Gaussian noise to add.
        :return: outputs with noise added.
        """
        return outputs + noise_level * torch.randn_like(outputs)
   

class HistoricDataLoader:
    
    '''
    Consideration of what needs to be transfered from the historic dataset to the new one:
    - input and output data
    - bounds
    - pareto front
    - optimization iteration
    - form of special setup (for latent variable model) 
    - there also needs to be an identifier regarding the initiating order of the historic datasets? 
           - what about the iterative adding off new data, but with each time the number of datapoints will be reduced, since knowledge is transferred
           - I probably need not only an order identifier, but also a "point in time" identifier, since t here can be multiple start datasets, ...
            - thought: point in time identifier not necessary, since the datapoints are just taken all as a whole and used to initialize the model. 

    Best solution will probably be to save it as a pickle file: dictionary structure can be used to save all the necessary information

    For stage 1 i will just save the datasets as points
    '''

    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype

    def load_dataset(self, hist_dataset: dict) -> Dataset:
        
        identifier = hist_dataset['identifier']
        input_data = hist_dataset['input_dataset']
        output_data = hist_dataset['output_dataset']
        bounds = hist_dataset['bounds']

        historic_dataset = Dataset(input_data=input_data, output_data=output_data, bounds=bounds, datamanager_type='historic', dtype=self.dtype, identifier=identifier)
        return historic_dataset