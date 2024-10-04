from botorch.test_functions.multi_objective import VehicleSafety
import torch


# function = VehicleSafety()

# #bounds: bounds = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0, 3.0]])

# #one five dimensional point as input

# input_x = [3.0, 3.0, 1.0, 2.116825545190717, 3.0]

# train_X_point = torch.tensor([input_x], dtype=torch.float64)

# y = function.evaluate_true(train_X_point)

# print(f"input: {train_X_point}")

# # Convert the tensor to a NumPy array
# y_values = y.numpy()

# # Check if y is a multi-dimensional array
# if y_values.ndim > 1:
#     # If it's multi-dimensional, iterate over each element in the nested arrays
#     formatted_y = [[f"{value:.4f}" for value in row] for row in y_values]
# else:
#     # If it's 1-dimensional, iterate directly
#     formatted_y = [f"{value:.4f}" for value in y_values]

# print(f"output: {formatted_y}")


from botorch.test_functions.multi_objective import BraninCurrin

function = BraninCurrin()

#bounds: bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

#one two dimensional point as input

input_x = [0.9467, 0.1016]

train_X_point = torch.tensor([input_x], dtype=torch.float64)

y = function.evaluate_true(train_X_point)

print(f"input: {train_X_point}")

# Convert the tensor to a NumPy array

y_values = y.numpy()

# Check if y is a multi-dimensional array
if y_values.ndim > 1:
    # If it's multi-dimensional, iterate over each element in the nested arrays
    formatted_y = [[f"{value:.4f}" for value in row] for row in y_values]
else:
    # If it's 1-dimensional, iterate directly
    formatted_y = [f"{value:.4f}" for value in y_values]

print(f"output: {formatted_y}")
