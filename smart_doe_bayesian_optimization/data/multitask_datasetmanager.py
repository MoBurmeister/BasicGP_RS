from data.create_dataset import DataManager
import numpy as np
import torch

#datasetmanager that specifically includes additional multitask model datasets

class MultiTaskDatasetManager:
    """
    A class to manage multi-task datasets for Bayesian optimization.
    Attributes:
    -----------
    dataset_manager : DataManager
        An instance of DataManager that provides access to the initial and historic datasets.
    task_index_list : list
        A list to keep track of task indices for each data point.
    train_X_taskdataset : torch.Tensor
        A tensor containing the input data with an additional task dimension.
    train_Y_single_taskdatasets : list of torch.Tensor
        A list of tensors, each containing the output data for a single task.
    Methods:
    --------
    __init__(dataset_manager: DataManager):
        Initializes the MultiTaskDatasetManager with the given DataManager instance.
    initiate_train_X_taskdataset():
        Prepares and returns the input data tensor with task dimensions.
    initiate_train_Y_single_taskdatasets():
        Prepares and returns a list of output data tensors, each corresponding to a single task.
    add_point_to_taskdatasets(new_X: torch.Tensor, new_Y: torch.Tensor):
        Adds a new data point to the task datasets.
    """
    
    def __init__(self, dataset_manager: DataManager):
        self.dataset_manager = dataset_manager
        self.task_index_list = None
        print("Initializing the train_X task dataset...")
        self.train_X_taskdataset = self.initiate_train_X_taskdataset()
        print("Initializing the train_Y single task datasets...")
        self.train_Y_single_taskdatasets = self.initiate_train_Y_single_taskdatasets()

    def initiate_train_X_taskdataset(self):
        """
        Prepares and concatenates the initial and historic datasets with an added task dimension.
        This method performs the following steps:
        1. Retrieves the initial and historic datasets from the dataset manager.
        2. Adds a task dimension to the initial dataset.
        3. Adds task dimensions to each of the historic datasets.
        4. Concatenates the initial and historic datasets into a single training dataset with task dimensions.
        5. Updates the task index list to reflect the task indices for each data point.
        Returns:
            torch.Tensor: A tensor containing the concatenated datasets with task dimensions.
        """
        
        self.task_index_list = []

        # Retrieve datasets (historic and initial)
        historic_datasets = self.dataset_manager.historic_dataset_list
        initial_dataset = self.dataset_manager.initial_dataset
        
        # Prepare the initial dataset with task dimension
        initial_data = initial_dataset.input_data
        if isinstance(initial_data, np.ndarray):
            initial_data = torch.tensor(initial_data, dtype=torch.float64)
        initial_task = torch.zeros((initial_data.shape[0], 1), dtype=torch.float64)
        initial_data_with_task = torch.cat((initial_task, initial_data), dim=1) 

        self.task_index_list.extend([0] * initial_data.shape[0])
        
        # Prepare historic datasets with task dimensions
        historic_data_with_tasks = []
        for i, dataset_dict in enumerate(historic_datasets):
            historic_data = dataset_dict['input_data']
            if isinstance(historic_data, np.ndarray):
                historic_data = torch.tensor(historic_data, dtype=torch.float64)
            task_number = torch.full((historic_data.shape[0], 1), i + 1, dtype=torch.float64)
            historic_data_with_task = torch.cat((task_number, historic_data), dim=1)
            historic_data_with_tasks.append(historic_data_with_task)

            self.task_index_list.extend([i + 1] * historic_data.shape[0])

        # Concatenate all datasets to create the new train_X
        all_data_with_tasks = [initial_data_with_task] + historic_data_with_tasks
        train_X_dataset = torch.cat(all_data_with_tasks, dim=0).to(torch.float64)

        # Summarized printout
        print(f"Train_X task dataset created.")
        print(f"Initial dataset: {initial_data.shape[0]} datapoints and shape {initial_data.shape}, Historic datasets: {len(historic_datasets)} with "
              f"total {sum([hd.shape[0] for hd in historic_data_with_tasks])} datapoints.")
        print(f"Final dataset shape: {train_X_dataset.shape} (including task dimension)")
        print(50*"-")

        return train_X_dataset
    
    def initiate_train_Y_single_taskdatasets(self):
        """
        Prepares and concatenates the output data from initial and historic datasets, 
        then splits the concatenated output into single-dimension datasets.
        This function performs the following steps:
        1. Retrieves the initial dataset and historic datasets from the dataset manager.
        2. Converts the output data of the initial dataset to a PyTorch tensor if it is a NumPy array.
        3. Converts the output data of each historic dataset to a PyTorch tensor if it is a NumPy array.
        4. Concatenates the output data from the initial and historic datasets.
        5. Splits the concatenated output data into single-dimension datasets.
        6. Prints a summary of the concatenated output shape and the number of single task datasets created.
        Returns:
            list of torch.Tensor: A list of single-dimension datasets, each represented as a PyTorch tensor.
        """
        # Retrieve datasets (historic and initial)
        historic_datasets = self.dataset_manager.historic_dataset_list
        initial_dataset = self.dataset_manager.initial_dataset
        
        # Prepare the initial dataset's output data
        initial_output = initial_dataset.output_data
        if isinstance(initial_output, np.ndarray):
            initial_output = torch.tensor(initial_output, dtype=torch.float64)
        
        # Prepare historic datasets' output data
        historic_outputs = []
        for i, dataset_dict in enumerate(historic_datasets):
            historic_output = dataset_dict['output_data']
            if isinstance(historic_output, np.ndarray):
                historic_output = torch.tensor(historic_output, dtype=torch.float64)
            historic_outputs.append(historic_output)
        
        # Concatenate all outputs (initial and historic) in the same order as X
        all_outputs = [initial_output] + historic_outputs
        concatenated_output = torch.cat(all_outputs, dim=0)
        
        # Split the concatenated output into single-dimension datasets
        num_outputs = concatenated_output.shape[1]
        single_task_datasets = [concatenated_output[:, i:i+1] for i in range(num_outputs)]
        
        # Summarized printout
        print(f"Train_Y single task datasets created.")
        print(f"Final concatenated output shape: {concatenated_output.shape}")
        print(f"Number of single task Y datasets: {len(single_task_datasets)}, each with shape {[single_task_datasets[0].shape]}")
        print(50*"-")

        return single_task_datasets
    
    def add_point_to_taskdatasets(self, new_X, new_Y):
        """
        Adds a new data point to the task datasets.
        Parameters:
        new_X (torch.Tensor): A tensor containing the new input data point with shape ([1, d]) and dtype torch.float64.
        new_Y (torch.Tensor): A tensor containing the new output data point with dtype torch.float64.
        Raises:
        ValueError: If new_X or new_Y are not torch tensors.
        ValueError: If new_X or new_Y do not have dtype torch.float64.
        ValueError: If new_X does not have the correct shape ([1, d]).
        This method performs the following steps:
        1. Ensures new_X and new_Y are torch tensors with dtype float64.
        2. Ensures new_X has the correct shape ([1, d]).
        3. Adds task index 0 for the new point to the task_index_list.
        4. Adds the new task index to new_X and concatenates it with train_X_taskdataset.
        5. Concatenates the new_Y with each single task dataset in train_Y_single_taskdatasets.
        """
        # Ensure new_X and new_Y are torch tensors with dtype float64
        if not isinstance(new_X, torch.Tensor) or not isinstance(new_Y, torch.Tensor):
            raise ValueError("new_X and new_Y must be torch tensors.")
        if new_X.dtype != torch.float64 or new_Y.dtype != torch.float64:
            raise ValueError("new_X and new_Y must have dtype torch.float64.")

        # Ensure new_X has the correct shape ([1, d])
        if new_X.dim() != 2 or new_X.shape[0] != 1:
            raise ValueError("new_X must have shape ([1, d])")

        # Add task index 0 for the new point to the task_index_list
        self.task_index_list.append(0)

        # Add the new task index to new_X and concatenate with train_X_taskdataset
        new_task_index = torch.zeros((1, 1), dtype=torch.float64)  # Task index 0
        new_X_with_task = torch.cat((new_task_index, new_X), dim=1)
        self.train_X_taskdataset = torch.cat((self.train_X_taskdataset, new_X_with_task), dim=0)

        # Concatenate the new_Y with each single task dataset in train_Y_single_taskdatasets
        for i in range(len(self.train_Y_single_taskdatasets)):
            self.train_Y_single_taskdatasets[i] = torch.cat((self.train_Y_single_taskdatasets[i], new_Y[:, i:i+1]), dim=0)