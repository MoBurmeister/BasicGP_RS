import torch


def check_dataset_shape(dataset: torch.Tensor) -> None:
    """
    Check if the dataset has the shape ([n, d]).

    Args:
        dataset (torch.Tensor): The dataset to be verified.

    Raises:
        ValueError: If the dataset is not two-dimensional.
    """
    if dataset.ndim != 2:
        raise ValueError(f"Dataset should be 2-dimensional, but got a {dataset.ndim}-dimensional tensor")
    

def check_dataset_same_size(input: torch.Tensor, output: torch.Tensor) -> None:
    """
    Check if the input and output datasets have the same number of rows (n).

    Args:
        input (torch.Tensor): The input dataset to be verified.
        output (torch.Tensor): The output dataset to be verified.

    Raises:
        ValueError: If the number of rows (n) in input and output datasets are not equal.
    """
    
    if input.shape[0] != output.shape[0]:
        raise ValueError(f"Number of rows (n) should be the same for both input and output. Got {input.shape[0]} rows in input and {output.shape[0]} rows in output")


def check_dataset_bounds(bounds: torch.Tensor, input_dim: int) -> None:
    """
    Check if the bounds of the dataset are correct.

    Args:
        bounds (torch.Tensor): The bounds to be verified.
        input_dim (int): The expected dimension (d) of the bounds.

    Raises:
        ValueError: If the bounds are not of shape (2, d) or if the input_dim is not equal to d.
    """
    if bounds.shape != (2, input_dim):
        raise ValueError(f"Bounds should be of shape (2, {input_dim}), but got {bounds.shape}")


def check_type(value, expected_type):
    """
    Checks if the value is of the expected type.

    Parameters:
    value (any): The value to check.
    expected_type (type): The expected type of the value.

    Raises:
    TypeError: If the value is not of the expected type.
    """
    if not isinstance(value, expected_type):
        raise TypeError(f"Expected type {expected_type.__name__}, got {type(value).__name__}")


def check_same_dimension(tensor1: torch.Tensor, tensor2: torch.Tensor) -> None:
    """
    Check if two tensors have the same number of dimensions.

    Args:
        tensor1 (torch.Tensor): The first tensor to be verified.
        tensor2 (torch.Tensor): The second tensor to be verified.

    Raises:
        ValueError: If the number of dimensions of the two tensors are not equal.
    """
    if tensor1.ndim != tensor2.ndim:
        raise ValueError(f"Number of dimensions should be the same for both tensors. Got {tensor1.ndim} dimensions in tensor1 and {tensor2.ndim} dimensions in tensor2")
