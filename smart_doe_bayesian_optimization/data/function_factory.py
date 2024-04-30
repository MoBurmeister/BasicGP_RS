import torch


'''
The inputs-vector is organized in a torch.Size([n, d]) fashion:

-n: number of datapoints
-d: number of dimensions.

-> so n datapoints where each has d dimensions

Therefore:
dataset_xsinx.func_create_dataset(xsinx.function_xsinx, num_datapoints=5, sampling_method="random", noise_level=0, x1_range=(0,6), x2_range=(5,10), x3_range=(100,200)) results in:
tensor([[  1.3362,   7.6596, 130.9058],
        [  4.9034,   9.9346, 112.4839],
        [  5.8313,   8.9086, 154.5993],
        [  3.8676,   5.4702, 156.1881],
        [  4.5060,   6.5246, 162.0561]])
torch.Size([5, 3])

'''

class FunctionFactory:
    @staticmethod
    def check_dimensions(inputs, expected_shape):
        """
        Checks if the input tensor matches the expected shape.
        Raises an exception if the shapes do not match.
        """
        if inputs.shape[1] != expected_shape:
            raise ValueError(f"Expected input shape to have {expected_shape} dimensions, but got {inputs.shape[1]}")
    
    @staticmethod
    def function_xsinx(inputs):
        
        expected_shape = 1

        FunctionFactory.check_dimensions(inputs=inputs, expected_shape=expected_shape)

        return inputs * torch.sin(inputs)
    
    @staticmethod
    def sum_of_sines(inputs):
        """
        Expects inputs tensor of shape [n, 3], where n is any number of data points.
        Applies sine to each dimension and sums the results.
        """
        FunctionFactory.check_dimensions(inputs, 3)
        result = torch.sin(inputs)
        return result.sum(dim=1)
    
    @staticmethod
    def multi_inputs(inputs):
        pass