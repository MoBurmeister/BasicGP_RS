import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
'''
The input-vector is organized in a torch.Size([n, d]) fashion:
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

This also holds true for the output, it needs to be in the shape of ([n, d])

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

        output = inputs * torch.sin(inputs)
        output = output.view(-1, 1)  # Reshape the output to have shape [n, 1]

        return output, expected_shape
    
    @staticmethod
    def sum_of_sines(inputs):
        """
        Expects inputs tensor of shape [n, 3], where n is any number of data points.
        Applies sine to each dimension and sums the results.
        """
        #3-dimensional input
        expected_input_shape = 3
        #1-dimensional output
        expected_output_shape = 1
        #here: three input ranges go into 3 sinuses (3-dim input) and then get add up to one-dim output

        FunctionFactory.check_dimensions(inputs, expected_input_shape)
        result = torch.sin(inputs)
        return result.sum(dim=1, keepdim=True), expected_output_shape
    
    @staticmethod
    def multi_inputs(inputs):
        # Define the expected output shape
        expected_output_shape = 3

        # Get the shape of the input tensor
        n, d = inputs.shape

        # Initialize the output tensor with shape (n, 3)
        outputs = torch.zeros((n, expected_output_shape))

        # Generate the shifted sinus curves
        for i in range(expected_output_shape):
            shift = i * math.pi / 2  # Different phase shifts (0, pi/2, pi)
            outputs[:, i] = torch.sin(inputs[:, 0] + shift)  # Apply the shift to the sinus curve

        return outputs, expected_output_shape
    
    @staticmethod
    def laser_heat_treatment(inputs):
        material_constants = {
            'lambda_th': 24.3,  # Example value
            'c_p': 389.36,       # Example value
            'alpha': 0.22,     # Example value
            'rho': 7800,      # Example value
            'T_Haerten': 1010+273 # Example value, target temperature for hardening
        }
        # Extract material constants from the dictionary
        lambda_th = material_constants['lambda_th']
        c_p = material_constants['c_p']
        alpha = material_constants['alpha']
        rho = material_constants['rho']
        T_Haerten = material_constants['T_Haerten']
               

        def Temp_fkt(x, y, z, laser_power, laser_speed, laser_width, alpha, lambda_th, rho, c_p, k, t):
            T_Integral = torch.exp(-(((x + laser_speed * t) ** 2 + y ** 2) / ((laser_width ** 2 / 8) + 4 * k * t) + (z ** 2 / (4 * k * t)))) / (torch.sqrt(t) * ((laser_width ** 2 / 8) + 4 * k * t))
            T_Int_Loesung = integrate.simpson(T_Integral.numpy(), t.numpy())
            T_Jarwitz = ((alpha * laser_power) / (np.pi ** 1.5 * np.sqrt(lambda_th * rho * c_p))) * T_Int_Loesung + 293
            return T_Jarwitz

        def simulate_laser_heating(laser_power, laser_speed, laser_width, lambda_th, c_p, alpha, rho, T_Haerten):
            resolution = 10000
            x_start, x_end, dx = -0.02, 0.01, 0.001
            y_start, y_end, dy = -0.005, 0.005, 0.001
            z = 0  # Surface temperature calculation
            t = torch.logspace(-6, 6, resolution, dtype=torch.float64)  # Time parameters

            k = lambda_th / (rho * c_p)

            T_start = 298  # Room temperature in K

            x_range = torch.arange(x_start, x_end, dx, dtype=torch.float64)
            y_range = torch.arange(y_start, y_end, dy, dtype=torch.float64)
            T = torch.zeros((len(x_range), len(y_range)), dtype=torch.float64)

            for i, x1 in enumerate(x_range):
                for j, y1 in enumerate(y_range):
                    T[i, j] = Temp_fkt(x1, y1, z, laser_power, laser_speed, laser_width, alpha, lambda_th, rho, c_p, k, t)

            T_verlauf = T[:, len(y_range) // 2]
            T_max = T.max().item() - 273

            hardening_time = sum(1 for k in range(len(x_range)) if T_verlauf[k] > T_Haerten) * (dx / laser_speed)
            
            return hardening_time, T_max

        results = []
        for params in inputs:
            laser_power, laser_speed, laser_width = params
            result = simulate_laser_heating(laser_power, laser_speed, laser_width, lambda_th, c_p, alpha, rho, T_Haerten)
            results.append(result)

        expected_output_shape = 2

        return torch.tensor(results, dtype=torch.float64), expected_output_shape

    
    # def laser_heat_treatment(inputs):

    #     '''
    #     inputs: 
    #     material constants: 
    #     - lambda_th: thermal conductivity
    #     - c_p: specific heat capacity
    #     - alpha: absorption coefficient
    #     - rho: density
    #     (- k: thermal diffusivity) -> k = lambda_th / (rho * c_p)
    #     - T_Haerten: target temperature for hardening process

    #     laser parameters (these can be variated):
    #     - laser_power: laser power
    #     - laser_speed: laser speed 
    #     - laser_width: laser width
    #     '''
    #     material_constants = {
    #         'lambda_th': 24.3,  # Example value
    #         'c_p': 389.36,       # Example value
    #         'alpha': 0.22,     # Example value
    #         'rho': 7800,      # Example value
    #         'T_Haerten': 1010+273 # Example value, target temperature for hardening
    #     }
    

    #     def Temp_fkt(x, y, z, laser_power, laser_speed, laser_width, alpha, lambda_th, rho, c_p, k, t):
    #         T_Integral = np.exp(-(((x + laser_speed * t) ** 2 + y ** 2) / ((laser_width ** 2 / 8) + 4 * k * t) + (z ** 2 / (4 * k * t)))) / (np.sqrt(t) * ((laser_width ** 2 / 8) + 4 * k * t))
    #         T_Int_Loesung = integrate.simpson(T_Integral, t)
    #         T_Jarwitz = ((alpha * laser_power) / (np.pi ** 1.5 * np.sqrt(lambda_th * rho * c_p))) * T_Int_Loesung + 293
    #         return T_Jarwitz


    #     def simulate_laser_heating(laser_power, laser_speed, laser_width, lambda_th, c_p, alpha, rho, T_Haerten):
            
    #         #def simulation parameters
    #         resolution = 10000
    #         x_start, x_end, dx = -0.02, 0.01, 0.001
    #         y_start, y_end, dy = -0.005, 0.005, 0.001
    #         z = 0 # hier kann eingestellt werden, ob Temperatur an Oberfläche oder in bestimmten Abstand berechnet werden soll
    #         t = np.logspace(-6, 6, resolution) # Zeitliche Parameter

    #         #material constants:
    #         k = lambda_th / (rho * c_p)

    #         #temp parameter: this is not used so far, can prob be deleted
    #         T_start = 298 # Raumtemperatur in K

    #         #initialize temperature field
    #         T = np.zeros((int(abs((x_start - x_end) / dx)), int(abs((y_start - y_end) / dy))))

    #         x1, y1 = x_start, y_start
    #         time_vector = []
    #         i = j = 0

    #         while x1 < x_end:
    #             y1 = y_start
    #             while y1 < y_end:
    #                 T[i][j] = Temp_fkt(x1, y1, z, laser_power, laser_speed, laser_width, alpha, lambda_th, rho, c_p, k, t)
    #                 y1 += dy
    #                 j += 1
    #             i += 1
    #             time_vector.append(x1 / laser_speed)
    #             x1 += dx
    #             j = 0

    #         T_verlauf = T[:, int(abs((y_start - y_end) / dy) / 2)]
    #         print("T_max = ", T.max() - 273, "°C")

    #         T_Haerten = sum(1 for k in range(int(abs((x_start - x_end) / dx))) if T_verlauf[k] > T_Haerten) * (dx / laser_speed)
    #         print("t (Härten) =", T_Haerten)

    #         #Two targets as an output: max holding time and min temperature difference between target and actual temperature
    #         return T_Haerten, T.max() - 273
        
              
    #     expected_output_shape = 2

    #     # Extract material constants from the dictionary
    #     lambda_th = material_constants['lambda_th']
    #     c_p = material_constants['c_p']
    #     alpha = material_constants['alpha']
    #     rho = material_constants['rho']
    #     T_Haerten_target = material_constants['T_Haerten']

    #     # Initialize the output array
    #     results = np.zeros((len(inputs), 2))

    #     # Iterate over each set of input parameters and perform the simulation
    #     for i, params in enumerate(inputs):
    #         laser_power, laser_speed, laser_width = params
    #         T_Haerten, T_max = simulate_laser_heating(laser_power, laser_speed, laser_width, lambda_th, c_p, alpha, rho, T_Haerten_target)
    #         results[i] = [T_Haerten, T_max]

    #     return results, expected_output_shape



