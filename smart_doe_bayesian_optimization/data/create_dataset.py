import torch
from typing import Callable, List, Tuple
from utils.checking_utils import check_type, check_dataset_shape, check_dataset_same_size
from data.dataset import Dataset
import pickle
import os
import numpy as np
from scipy.stats import qmc

class DataManager:

    def __init__(self, external_input: bool, dataset_func: Callable = None, dtype: torch.dtype = torch.float64):
        self.historic_data_loader = HistoricDataLoader(dtype=dtype)
        self.initial_data_loader = InitialDataLoader(dtype=dtype)
        self.dataset_func = dataset_func
        self.external_input = external_input
        self.input_dim = None
        self.output_dim = None
        self.initial_dataset = None
        self.input_parameter_name = None
        self.output_parameter_name = None
        #init continues below
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
        self.maximization_flags = []
    
    def load_initial_dataset(self, num_datapoints:int, bounds: List[tuple], maximization_flags: List[bool], input_parameter_name: List[str], output_parameter_name: List[str], sampling_method: str = "grid", noise_level: float = 0.0, identifier:int=None):
        '''
        If a flag is False, it indicates the corresponding output dimension should be minimized!
        '''
        self.input_parameter_name = input_parameter_name
        self.output_parameter_name = output_parameter_name

        if self.external_input:
            initial_dataset = self.initial_data_loader.create_inital_dataset_manually(num_datapoints, bounds, maximization_flags, self.input_parameter_name, self.output_parameter_name, identifier=identifier)
        else:
            initial_dataset = self.initial_data_loader.load_dataset(self.dataset_func, num_datapoints, bounds, maximization_flags, sampling_method, noise_level, identifier)
        check_type(initial_dataset, Dataset)
        self.initial_dataset = initial_dataset
        self.set_check_input_output_dim(initial_dataset.input_dim, initial_dataset.output_dim)
        self.set_check_maximization_flags(dataset=initial_dataset)
        
        print("Initial dataset added (and old one discarded):")
        print(f"Number of datapoints: {self.initial_dataset.input_data.shape[0]}")
        print(f"Number of input dimensions: {self.initial_dataset.input_data.shape[1]}")
        print(f"Number of output dimensions: {self.initial_dataset.output_data.shape[1]}")
        print('-' * 50)

    def load_historic_dataset(self, dataset_path: str):
        # each historic dataset itself carries a one model per objective? The model is then defined by cov module and mean? Or more the cov matrix
        # the historic datasets should rather probably just carry the data points, bounds etc. Initializ model here? 
        # Load the historic dataset from the pickle file

        historic_dict_list = []

        # Iterate through all files in the directory
        for filename in os.listdir(dataset_path):
            if filename.endswith('.pkl') or filename.endswith('.pickle'):
                file_path = os.path.join(dataset_path, filename)
                with open(file_path, 'rb') as f:
                    dataset = pickle.load(f)
                    historic_dict_list.append(dataset)

        for hist_dataset in historic_dict_list:
            transformed_dataset = self.historic_data_loader.load_dataset(hist_dataset)
            self.set_check_input_output_dim(transformed_dataset.input_dim, transformed_dataset.output_dim)
            self.historic_datasets.append(transformed_dataset)
            self.set_check_maximization_flags(dataset=transformed_dataset)
            print(f"Historic dataset added:")
            print(f"Identifier: {transformed_dataset.identifier}")
            print(f"Number of datapoints: {transformed_dataset.input_data.shape[0]}")
            print(f"Number of input dimensions: {transformed_dataset.input_data.shape[1]}")
            print(f"Number of output dimensions: {transformed_dataset.output_data.shape[1]}")
            print('-' * 50)

    def add_point_to_initial_dataset(self, point: Tuple[torch.Tensor, torch.Tensor]):
        #Add a single point to the initial dataset
        #Should only set one point of shape ([1, d])
        if self.initial_dataset is None:
            raise ValueError("No initial dataset set. Please load an initial dataset first.")
        self.set_check_input_output_dim(point[0].shape[1], point[1].shape[1])
        check_dataset_shape(point[0])
        check_dataset_shape(point[1])
        self.initial_dataset.input_data = torch.cat([self.initial_dataset.input_data, point[0]], dim=0)
        self.initial_dataset.output_data = torch.cat([self.initial_dataset.output_data, point[1]], dim=0)
        check_dataset_same_size(self.initial_dataset.input_data, self.initial_dataset.output_data)


    def set_check_input_output_dim(self, input_dim: int, output_dim: int):
        '''
        Function checks if the input and output dimensions match the existing dimensions. If not, it raises a ValueError.
        If no dimensions are set yet, it sets the dimensions to the provided values.
        '''
        if self.input_dim is not None and self.output_dim is not None:
            if self.input_dim != input_dim or self.output_dim != output_dim:
                raise ValueError("Input and output dimensions must match the existing dimensions.")
        else:
            self.input_dim = input_dim
            self.output_dim = output_dim

    def set_check_maximization_flags(self, dataset: Dataset):
        if self.maximization_flags:
            if self.maximization_flags != dataset.maximization_flags:
                raise ValueError("maximization flags must match the existing flags in the DatasetManager.")
        else:
            self.maximization_flags = dataset.maximization_flags
            print(f"maximization flags in DatasetManager set to: {self.maximization_flags}")
            print(50*"-")
            

    
class InitialDataLoader:

    #here without classmethod, since there can be the case that I need costum configs for these initiated loaders 

    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype

    def load_dataset(self, dataset_func: Callable[..., torch.Tensor], num_datapoints:int, bounds: List[tuple], maximization_flags: List[bool], sampling_method: str = "grid", noise_level: float = 0.0, identifier: int =None) -> Dataset:
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
        num_dimensions = len(bounds)

        if sampling_method not in ['random', 'grid', 'LHS']:
            raise ValueError("Sampling method must be 'random', 'grid' or 'LHS'.")

        print(f"Loading initial dataset with {num_datapoints}, from literature {11*num_dimensions-1} datapoints are recommended!")
        
        inputs = []

        if num_datapoints == 0:
            inputs = torch.tensor([], dtype=torch.float64).reshape(0, num_dimensions)
            output = torch.tensor([], dtype=torch.float64).reshape(0, len(maximization_flags))
        else:
            if sampling_method == "random":
                for touple in bounds:
                    inputs.append(torch.rand(num_datapoints) * (touple[1] - touple[0]) + touple[0])
                inputs = torch.stack(inputs, dim=1)
            elif sampling_method == "grid":
                for touple in bounds:
                    inputs.append(torch.linspace(touple[0], touple[1], steps=num_datapoints))
                inputs = torch.stack(inputs, dim=1)
            elif sampling_method == "LHS":
                sampler = qmc.LatinHypercube(d=num_dimensions)
                samples = sampler.random(n=num_datapoints)
                lower_bounds, upper_bounds = zip(*bounds)
                scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)
                inputs = torch.tensor(scaled_samples)

            output, _ = dataset_func(inputs)

            if noise_level > 0:
                output = self.add_noise(outputs=output, noise_level=noise_level)

        bounds_tensor = self.convert_bounds_to_tensor(bounds)        

        initial_datset = Dataset(input_data=inputs, output_data=output, bounds=bounds_tensor, datamanager_type="initial", maximization_flags=maximization_flags, identifier=identifier)

        return initial_datset
    

    def create_inital_dataset_manually(self, num_datapoints:int, bounds: List[tuple], maximization_flags: List[bool], input_parameter_name: List[str], output_parameter_name: List[str], identifier: int=None):

        #just LHS supported due to scientific literature

        print(f"Dataset is created manually based on LHS sampling.")

        num_datapoints = num_datapoints
        num_dimensions = len(bounds)
        num_outputs = len(maximization_flags)

        print(f"Loading initial dataset with {num_datapoints}, from literature {11*num_dimensions-1} datapoints are recommended!")

        sampler = qmc.LatinHypercube(d=num_dimensions)
        samples = sampler.random(n=num_datapoints)
        lower_bounds, upper_bounds = zip(*bounds)
        scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)
        inputs = torch.tensor(scaled_samples)

        print(f"Input values are: {inputs}")

        bounds_tensor = self.convert_bounds_to_tensor(bounds)

        # Collect user inputs for each input point
        outputs_list = []
        for i, input_point in enumerate(inputs):
            print(f"Input point {i + 1}/{num_datapoints}: {input_point.tolist()}")
            output_values = []
            for j in range(num_outputs):
                output_value = float(input(f"Enter the output value for the objective {output_parameter_name[j]} for this input point: "))
                output_values.append(output_value)
            outputs_list.append(output_values)


        # Convert the collected outputs into a tensor
        outputs = torch.tensor(outputs_list).reshape(num_datapoints, num_outputs)

        initial_dataset = Dataset(input_data=inputs, output_data=outputs, bounds=bounds_tensor, datamanager_type="initial", maximization_flags=maximization_flags, identifier=identifier)

        return initial_dataset

    
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
    # TODO: Transfer Learning of the historic model with its parameters/data

    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype

    def load_dataset(self, hist_dataset: dict) -> Dataset:
        
        identifier = hist_dataset['identifier']
        input_data = hist_dataset['input_dataset']
        output_data = hist_dataset['output_dataset']
        bounds = hist_dataset['bounds']
        maximization_flags = hist_dataset['maximization_flags']

        historic_dataset = Dataset(input_data=input_data, output_data=output_data, bounds=bounds, datamanager_type='historic', maximization_flags=maximization_flags, dtype=self.dtype, identifier=identifier)
        return historic_dataset