import torch

class FunctionFactory:
    @staticmethod
    def function_xsinx(inputs):

        # TODO: dimension check?

        return inputs * torch.sin(inputs)