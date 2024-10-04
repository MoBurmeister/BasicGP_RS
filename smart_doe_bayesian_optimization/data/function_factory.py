import torch
from torch import Tensor
import math
import numpy as np
from scipy import integrate
from botorch.test_functions.multi_objective import VehicleSafety
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

    def __init__(self, variation_factor=0.0):
        self.variation_factor = variation_factor

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
            x_start, x_end, dx = -0.0005, 0.0005, 0.00001
            y_start, y_end, dy = -0.0005, 0.0005, 0.00001
            z = 0  # Surface temperature calculation
            t = torch.logspace(-6, 6, resolution, dtype=torch.float64)  # Time parameters

            k = lambda_th / (rho * c_p)

            T_start = 298  # Room temperature in K

            x_range = torch.arange(x_start, x_end, dx, dtype=torch.float64)
            y_range = torch.arange(y_start, y_end, dy, dtype=torch.float64)
            T = torch.zeros((len(x_range), len(y_range)), dtype=torch.float64)

            for i, x1 in enumerate(x_range):
                for j, y1 in enumerate(y_range):

                    T[i, j] = T_start + Temp_fkt(x1, y1, z, laser_power, laser_speed, laser_width, alpha, lambda_th, rho, c_p, k, t)

            T_verlauf = T[:, len(y_range) // 2]
            T_max = T.max().item() 

            t_max_allowed = 1010 + 40 + 273

            t_diff = abs(T_max - t_max_allowed)

            hardening_time = sum(1 for k in range(len(x_range)) if T_verlauf[k] > T_Haerten) * (dx / laser_speed)
            
            #print(T_max, t_diff, t_max_allowed)

            return hardening_time, t_diff

        #maximize hardening_time while minimizing t_max but keep it below threshold

        results = []
        for params in inputs:
            laser_power, laser_speed, laser_width = params
            result = simulate_laser_heating(laser_power, laser_speed, laser_width, lambda_th, c_p, alpha, rho, T_Haerten)
            results.append(result)

        expected_output_shape = 2

        return torch.tensor(results, dtype=torch.float64), expected_output_shape
    

    @staticmethod

    #_max_hv = 59.36011874867746  # this is approximated using NSGA-II

    def BraninCurrin(inputs):
        def evaluate_true(X: Tensor) -> Tensor:
            # Rescale inputs for Branin
            x0 = 15 * X[..., 0] - 5
            x1 = 15 * X[..., 1]
            
            # Branin function (using the rescaled x0, x1)
            f1 = (
                (x1 - 5.1 * x0 ** 2 / (4 * math.pi ** 2) + 5 * x0 / math.pi - 6) ** 2
                + 10 * (1 - 1 / (8 * math.pi)) * torch.cos(x0) + 10
            )
            
            # Currin function (no change needed)
            f2 = (
                (1 - torch.exp(-1 / (2 * X[..., 1]))) * (
                    2300 * X[..., 0] ** 3 + 1900 * X[..., 0] ** 2 + 2092 * X[..., 0] + 60
                ) / (100 * X[..., 0] ** 3 + 500 * X[..., 0] ** 2 + 4 * X[..., 0] + 20)
            )
            return torch.stack([f1, f2], dim=-1)
        
        expected_output_shape = 2

        return evaluate_true(inputs), expected_output_shape
            
    
    @staticmethod
    def welding_beam(inputs):
        '''
        additional information: 
        _bounds = [
        (0.125, 5.0),
        (0.1, 10.0),
        (0.1, 10.0),
        (0.125, 5.0),
        ]
        _ref_point = [40, 0.015]
        '''

        def evaluate_true(X: Tensor) -> Tensor:
            x1, x2, x3, x4 = X.unbind(-1)
            f1 = 1.10471 * (x1**2) * x2 + 0.04811 * x3 * x4 * (14.0 + x2)
            f2 = 2.1952 / (x4 * x3**3)
            return torch.stack([f1, f2], dim=-1)

        expected_output_shape = 2
        
        return evaluate_true(inputs), expected_output_shape
    

    @staticmethod
    def vehicle_safety_design(inputs):

        def evaluate_true(self, X: Tensor) -> Tensor:
            X1, X2, X3, X4, X5 = torch.split(X, 1, -1)
            f1 = (
                1640.2823
                + 2.3573285 * X1
                + 2.3220035 * X2
                + 4.5688768 * X3
                + 7.7213633 * X4
                + 4.4559504 * X5
            )
            f2 = (
                6.5856
                + 1.15 * X1
                - 1.0427 * X2
                + 0.9738 * X3
                + 0.8364 * X4
                - 0.3695 * X1 * X4
                + 0.0861 * X1 * X5
                + 0.3628 * X2 * X4
                - 0.1106 * X1.pow(2)
                - 0.3437 * X3.pow(2)
                + 0.1764 * X4.pow(2)
            )
            f3 = (
                -0.0551
                + 0.0181 * X1
                + 0.1024 * X2
                + 0.0421 * X3
                - 0.0073 * X1 * X2
                + 0.024 * X2 * X3
                - 0.0118 * X2 * X4
                - 0.0204 * X3 * X4
                - 0.008 * X3 * X5
                - 0.0241 * X2.pow(2)
                + 0.0109 * X4.pow(2)
            )
            f_X = torch.cat([f1, f2, f3], dim=-1)
            return f_X

        #max hypervolume from paper: 246

        outputs = evaluate_true(inputs)

        expected_output_shape = 3

        return outputs, expected_output_shape
    
    def generate_car_crash_synthetic_data(self, outputs):
        '''
        Generate synthetic data for the vehicle safety design function.
        Variation factor is used to introduce noise to the data. It is a factor that is multiplied with the coefficients of the function.
        '''
        def evaluate_true(X: Tensor, coeffs) -> Tensor:
            X1, X2, X3, X4, X5 = torch.split(X, 1, -1)
            f1 = (
                coeffs[0] 
                + coeffs[1] * X1
                + coeffs[2] * X2
                + coeffs[3] * X3
                + coeffs[4] * X4
                + coeffs[5] * X5
            )
            f2 = (
                coeffs[6]
                + coeffs[7] * X1
                - coeffs[8] * X2
                + coeffs[9] * X3
                + coeffs[10] * X4
                - coeffs[11] * X1 * X4
                + coeffs[12] * X1 * X5
                + coeffs[13] * X2 * X4
                - coeffs[14] * X1.pow(2)
                - coeffs[15] * X3.pow(2)
                + coeffs[16] * X4.pow(2)
            )
            f3 = (
                coeffs[17]
                + coeffs[18] * X1
                + coeffs[19] * X2
                + coeffs[20] * X3
                - coeffs[21] * X1 * X2
                + coeffs[22] * X2 * X3
                - coeffs[23] * X2 * X4
                - coeffs[24] * X3 * X4
                - coeffs[25] * X3 * X5
                - coeffs[26] * X2.pow(2)
                + coeffs[27] * X4.pow(2)
            )
            f_X = torch.cat([f1, f2, f3], dim=-1)
            return f_X

        base_coeffs = [
            1640.2823, 2.3573285, 2.3220035, 4.5688768, 7.7213633, 4.4559504,
            6.5856, 1.15, 1.0427, 0.9738, 0.8364, 0.3695, 0.0861, 0.3628, 0.1106,
            0.3437, 0.1764, -0.0551, 0.0181, 0.1024, 0.0421, 0.0073, 0.024,
            0.0118, 0.0204, 0.008, 0.0241, 0.0109
        ]
        
        # Apply variation to coefficients
        varied_coeffs = [coeff * (1 + self.variation_factor) for coeff in base_coeffs]
        
        outputs = evaluate_true(outputs, varied_coeffs)
        expected_output_shape = 3
        return outputs, expected_output_shape

    
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



