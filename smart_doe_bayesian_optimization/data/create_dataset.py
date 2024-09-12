import torch
from typing import Callable, List, Tuple
from utils.checking_utils import check_type, check_dataset_shape, check_dataset_same_size
from data.dataset import Dataset
import pickle
import os
import numpy as np
from scipy.stats import qmc
import pandas as pd

class DataManager:

    def __init__(self, external_input: bool, dataset_func: Callable = None, variation_factor: float = None, historic_data_path: str = None, dtype: torch.dtype = torch.float64):
        self.historic_data_path = historic_data_path
        self.historic_data_loader = HistoricDataLoader(historic_data_path = self.historic_data_path, dtype=dtype)
        self.initial_data_loader = InitialDataLoader(dtype=dtype)
        self.dataset_func = dataset_func
        self.external_input = external_input
        self.variation_factor = variation_factor
        self.input_dim = None
        self.output_dim = None
        self.initial_dataset = None
        self.input_parameter_name = None
        self.output_parameter_name = None
        self.meta_data_dict = None
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

        self.historic_modelinfo_list = []
        self.historic_datasetinfo_list = []
        self.historic_dataset_list = []
        self.dtype = dtype
        self.maximization_flags = []
    
    def load_initial_dataset(self, num_datapoints:int, bounds: List[tuple], maximization_flags: List[bool], input_parameter_name: List[str], output_parameter_name: List[str], meta_data_dict: dict, sampling_method: str = "grid", noise_level: float = 0.0, identifier:int=None):
        '''
        If a flag is False, it indicates the corresponding output dimension should be minimized!
        '''
        self.input_parameter_name = input_parameter_name
        self.output_parameter_name = output_parameter_name

        #check that the sampling method is valid
        if sampling_method not in ['random', 'grid', 'LHS', 'from_file']:
            raise ValueError("Sampling method must be 'random', 'grid', 'LHS' or 'from_file'.")

        # check, that number of tuples in bounds matches number of input parameter names
        if len(bounds) != len(input_parameter_name):
            raise ValueError("Number of bounds must match the number of input parameter names. Equal number of input parameters!")
        
        # check, that number of maximization flags matches number of output parameter names 
        if len(maximization_flags) != len(output_parameter_name):
            raise ValueError("Number of maximization flags must match the number of output parameter names. Equal number of output objectives!")

        #must be more than 0 datapoints
        if num_datapoints == 0:
            raise ValueError("Number of datapoints must be greater than 0.")

        #check meta_data_dict

        # Check if meta_data_dict is a non-empty dictionary
        if not isinstance(meta_data_dict, dict) or len(meta_data_dict) == 0:
            raise ValueError("meta_data_dict must be a non-empty dictionary.")
        
        # Validate that all values in meta_data_dict are either float or int
        for key, value in meta_data_dict.items():
            if not isinstance(value, (float, int)):
                raise ValueError(f"The value for '{key}' in meta_data_dict must be an int or float, but got {type(value).__name__}.")
            
        #save meta data dict
        self.meta_data_dict = meta_data_dict
            
        # print meta_data_dict
        print("Meta data dictionary of initial dataset:")
        for key, value in meta_data_dict.items():
            print(f"  {key}: {value}")

        if self.external_input:
            # since in literature basicaly only LHS is recommended, it is only implemented here for manual dataset sampling. Besides this a complete manual data input is allowed
            initial_dataset = self.initial_data_loader.create_inital_dataset_manually(num_datapoints, bounds, maximization_flags, self.input_parameter_name, self.output_parameter_name, meta_data_dict, sampling_method, identifier=identifier)
        else:
            #check that a dataset function is provided
            if self.dataset_func is None:
                raise ValueError("No dataset function provided. Please provide a dataset function to load the initial dataset.")
            initial_dataset = self.initial_data_loader.load_dataset(self.dataset_func, num_datapoints, bounds, maximization_flags, meta_data_dict, sampling_method, noise_level, identifier)
        check_type(initial_dataset, Dataset)

        #only multi-objective is allowed here (otherwise problems with the code, which is setup for multi-objective)
        if initial_dataset.output_dim == 1:
            raise ValueError("Only multi-objective optimization is supported. Please provide more than one output dimension.")

        self.initial_dataset = initial_dataset
        self.set_check_input_output_dim(initial_dataset.input_dim, initial_dataset.output_dim)
        self.set_check_maximization_flags(dataset=initial_dataset)
        
        print("Initial dataset added (and old one discarded):")
        print(f"Number of datapoints: {self.initial_dataset.input_data.shape[0]}")
        print(f"Number of input dimensions: {self.initial_dataset.input_data.shape[1]}")
        print(f"Number of output dimensions: {self.initial_dataset.output_data.shape[1]}")
        print('-' * 50)

    def load_historic_data(self):

        #there needs to be an initial dataset setup:
        if self.initial_dataset is None:
            raise ValueError("No initial dataset set. Please load an initial dataset first.")
        
        if self.historic_data_path is None:
            raise ValueError("No historic data path provided. Please provide a path to load historic data.")

        #loads the three dicts from all historic optimization problems
        model_info_list, dataset_info_list, dataset_list = self.historic_data_loader.load_modeldata()

        self.historic_modelinfo_list = model_info_list
        self.historic_datasetinfo_list = dataset_info_list
        self.historic_dataset_list = dataset_list

        for historic_dataset in self.historic_datasetinfo_list:
            #print(historic_dataset)
            meta_data_dict = historic_dataset['metafeatures']
            #check that the structure of the meta data dict is the same as the inital dataset but do not need to be the same values
            if meta_data_dict.keys() != self.meta_data_dict.keys():
                raise ValueError("Meta data dictionary of historic dataset does not match the meta data dictionary of the initial dataset.")

        for historic_dataset in self.historic_datasetinfo_list:
            # print out meta data dict
            print("Meta data dictionary: of one historic dataset:")
            for key, value in historic_dataset['metafeatures'].items():
                print(f"  {key}: {value}")
        

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
        self.initial_dataset.num_datapoints += 1    
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

    def load_dataset(self, dataset_func: Callable[..., torch.Tensor], num_datapoints:int, bounds: List[tuple], maximization_flags: List[bool], meta_data_dict: dict, sampling_method: str = "grid", noise_level: float = 0.0, identifier: int =None) -> Dataset:
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
            raise ValueError("Sampling method must be 'random', 'grid' or 'LHS'. Everything else and from_file input is not supported here!")

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

        initial_datset = Dataset(input_data=inputs, output_data=output, bounds=bounds_tensor, datamanager_type="initial", maximization_flags=maximization_flags, meta_data_dict=meta_data_dict, identifier=identifier)

        return initial_datset
    

    def create_inital_dataset_manually(self, num_datapoints:int, bounds: List[tuple], maximization_flags: List[bool], input_parameter_name: List[str], output_parameter_name: List[str], meta_data_dict: dict, sampling_method: str, identifier: int=None):
        
        #check that the sampling method is valid
        if sampling_method not in ['LHS', 'from_file']:
            raise ValueError("Sampling method must be'LHS' or 'from_file'. Grid and random are not supported for manual input here!")
        
        num_datapoints = num_datapoints
        num_dimensions = len(bounds)
        num_outputs = len(maximization_flags)
    
        if sampling_method == 'LHS':

            #just LHS supported due to scientific literature

            print(f"Dataset is created manually based on LHS sampling.")

            print(f"Loading initial dataset with {num_datapoints}, from literature {11*num_dimensions-1} datapoints are recommended!")

            sampler = qmc.LatinHypercube(d=num_dimensions)
            samples = sampler.random(n=num_datapoints)
            lower_bounds, upper_bounds = zip(*bounds)
            scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)
            inputs = torch.tensor(scaled_samples)

            print(f"Input values are: {inputs} based on LHS sampling.")

            # Collect user inputs for each input point
            outputs_list = []
            for i, input_point in enumerate(inputs):
                while True:
                    print(f"\nInput point {i + 1}/{num_datapoints}: {input_point.tolist()}")
                    output_values = []
                    
                    for j in range(num_outputs):
                        while True:
                            try:
                                output_value = float(input(f"Enter the output value for the objective {j} '{output_parameter_name[j]}' for this input point: "))
                            except ValueError:
                                print("Invalid input. Please enter a numeric value.")
                                continue
                            
                            # Confirm the entered value
                            while True:
                                confirmation = input(f"You entered {output_value} for '{output_parameter_name[j]}'. Is this correct? (y/n): ").strip().lower()
                                if confirmation == 'y':
                                    output_values.append(output_value)
                                    break
                                elif confirmation == 'n':
                                    print("Let's try again.")
                                    break
                                else:
                                    print("Invalid input. Please enter 'y' for yes or 'n' for no.")
                            
                            # Break out of the outer loop if the value is confirmed
                            if confirmation == 'y':
                                break

                    # Confirm the entire input point
                    print("\nSummary of your input for this point:")
                    for j, value in enumerate(output_values):
                        print(f"  {output_parameter_name[j]}: {value}")
                    
                    final_confirmation = input("Is this entire input point correct? (y/n): ").strip().lower()
                    if final_confirmation == 'y':
                        outputs_list.append(output_values)
                        break
                    elif final_confirmation == 'n':
                        print("Let's start over for this input point.")
                    else:
                        print("Invalid input. Please enter 'y' for yes or 'n' for no.")

            # Convert the collected outputs into a tensor
            outputs = torch.tensor(outputs_list).reshape(num_datapoints, num_outputs)

            outputs = outputs.to(dtype=torch.float64)

        elif sampling_method == 'from_file':
            # Load data from Excel
            directory = 'smart_doe_bayesian_optimization/input_data_custom_initial_dataset/'
            excel_files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]
            
            if not excel_files:
                raise FileNotFoundError("No Excel file found in the directory.")
            
            file_path = os.path.join(directory, excel_files[0])
            df = pd.read_excel(file_path, sheet_name=0)

            # Assume the first num_dimensions columns are inputs and the remaining columns are outputs
            inputs_df = df.iloc[:, :num_dimensions]
            outputs_df = df.iloc[:, num_dimensions + 1:num_dimensions + 1 + num_outputs]

            # Convert the DataFrame to tensors
            inputs = torch.tensor(inputs_df.values, dtype=torch.float64)
            outputs = torch.tensor(outputs_df.values, dtype=torch.float64)

            #check if number of datapoints matches
            if inputs.shape[0] != num_datapoints:
                raise ValueError(f"Number of datapoints in the Excel file ({inputs.shape[0]}) does not match the provided number of datapoints ({num_datapoints}).")
            
            #check if number of input dimensions matches
            if inputs.shape[1] != num_dimensions:
                raise ValueError(f"Number of input dimensions in the Excel file ({inputs.shape[1]}) does not match the provided number of input dimensions ({num_dimensions}).")
            
            #check if number of output dimensions matches
            if outputs.shape[1] != num_outputs:
                raise ValueError(f"Number of output dimensions in the Excel file ({outputs.shape[1]}) does not match the provided number of output dimensions ({num_outputs}).")
            
            print(f"Input values are: {inputs} loaded from provided excel file. Shape is {inputs.shape}")
            print(f"Output values are: {outputs} loaded from provided excel file. Shape is {outputs.shape}")

        bounds_tensor = self.convert_bounds_to_tensor(bounds)

        initial_dataset = Dataset(input_data=inputs, output_data=outputs, bounds=bounds_tensor, datamanager_type="initial", maximization_flags=maximization_flags, meta_data_dict=meta_data_dict, identifier=identifier)

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
   

#The historic dataloader will load GPs from statedict historic data. It will also provide 

class HistoricDataLoader:
    
    '''
    OLD APPROACH (still here for information):
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
    '''
    def __init__(self, historic_data_path: str, dtype: torch.dtype = torch.float64):
        self.historic_data_path = historic_data_path
        self.dtype = dtype

    def load_modeldata(self):
        model_info_list = []
        dataset_info_list = []
        dataset_list = []

        # Iterate over all files in the directory
        for filename in os.listdir(self.historic_data_path):
            if filename.endswith(".pkl"):  # Only process pickle files
                file_path = os.path.join(self.historic_data_path, filename)
                
                # Load the pickle file
                with open(file_path, 'rb') as file:
                    combined_dict = pickle.load(file)
                    
                    # Extract the dictionaries and append to the respective lists
                    model_info_list.append(combined_dict['model_info'])
                    dataset_info_list.append(combined_dict['dataset_info'])
                    dataset_list.append(combined_dict['dataset'])

        return model_info_list, dataset_info_list, dataset_list
