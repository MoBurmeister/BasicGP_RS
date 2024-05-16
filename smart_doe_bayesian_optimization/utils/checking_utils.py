import torch

# TODO: implement functions here to check dimensions
# TODO: implement functions here to check similar size 
# TODO: implement function to check further similar similarity

def check_same_dimension(tensor1: torch.Tensor, tensor2: torch.Tensor) -> None:
    """
    Check if two tensors have the same dimensions.

    Args:
        tensor1 (torch.Tensor): The first tensor.
        tensor2 (torch.Tensor): The second tensor.

    Raises:
        ValueError: If the tensors do not have the same dimensions.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"The dimensions of the tensors are not the same. "
                         f"Tensor1 has shape {tensor1.shape} while Tensor2 has shape {tensor2.shape}.")
