import torch
from utils.checking_utils import check_dataset_shape, check_dataset_same_size, check_dataset_bounds


class Dataset:
    """
    A class representing a dataset. This is used in the inital dataset manageer and in the historic dataset manager.

    Data (input and output) must be of shape ([n, d]) where n is the number of datapoints and d is the number of dimensions.
    n of the input data must match n of the output data.

    Bounds must be of shape torch.Size([2, d]) - ([min_x1, min_x2, ...],[max_x1, max_x2, ...]). d must match the number of dimensions in the data.
    
    Attributes:
        input_data (torch.Tensor): The input data for the dataset.
        output_data (torch.Tensor): The output data for the dataset.
        bounds_list (torch.Tensor): The bounds list for the dataset.
        num_datapoints (int): The number of datapoints in the dataset.
        input_dim (int): The number of dimensions in the input data.
        output_dim (int): The number of dimensions in the output data.
    """

    def __init__(self, input_data: torch.Tensor, output_data: torch.Tensor, bounds: torch.Tensor, datamanager_type: str, dtype: torch.dtype = torch.float64, identifier: int = None):
        """
        Initializes a Dataset object.

        Args:
            input_data (torch.Tensor): The input data tensor.
            output_data (torch.Tensor): The output data tensor.
            bounds (torch.Tensor): The bounds tensor.
            dtype (torch.dtype, optional): The data type of the tensors. Defaults to torch.float64.
        """

        valid_datamanager_types = ["historic", "initial"]
        if datamanager_type not in valid_datamanager_types:
            raise ValueError(f"datamanager_type must be one of {valid_datamanager_types}, but got '{datamanager_type}'.")

        self.datamanager_type = datamanager_type

        self.identifier = identifier

        self.dtype = dtype

        check_dataset_shape(input_data)
        check_dataset_shape(output_data)
        check_dataset_same_size(input_data, output_data)

        self.input_data = input_data
        self.output_data = output_data

        self.num_datapoints = input_data.shape[0]
        self.input_dim = input_data.shape[1]
        self.output_dim = output_data.shape[1]    

        self.adjust_dtypes()
        
        check_dataset_bounds(bounds=bounds, input_dim=self.input_dim)
        self.bounds_list = bounds

        # if self.datamanager_type == "historic":
        #     if historic_data is None:
        #         raise ValueError("historic_data must be provided when datamanager_type is 'historic'.")
        #     self.historical_data = {
        #     "identifier": historic_data["identifier"],
        #     "mean_module": historic_data["mean_module"],
        #     "cov_module": historic_data["cov_module"]
        #     }


        print(f"Dataset created for {datamanager_type} with {self.input_dim} input dimensions and {self.output_dim} output dimensions.")
        print(f"Number of datapoints: {self.num_datapoints}")
        print(f"Bounds: {self.bounds_list.tolist()}")
        print('-' * 50)


    def adjust_dtypes(self):
        """
        Adjusts the data types of the input and output data to the specified dtype.

        This method converts the input and output data to the specified dtype.
        """
        self.input_data = self.input_data.to(self.dtype)
        self.output_data = self.output_data.to(self.dtype)