from botorch.utils.multi_objective.hypervolume import Hypervolume
import torch

def calculate_hypervolume_for_points(points: list[tuple], reference_point: tuple) -> list[float]:
    """
    Calculate the hypervolume for a list of points against a user-defined reference point.
    
    Args:
        points (list[tuple]): A list of tuples representing the points ((x1, y1), (x2, y2), ...).
        reference_point (tuple): A tuple representing the reference point (e.g., (ref_x, ref_y)).
        
    Returns:
        list[float]: A list of hypervolume values for each set of points.
    """
    # Convert input points and reference point to torch tensors
    points_tensor = torch.tensor(points, dtype=torch.float64)
    reference_point_tensor = torch.tensor(reference_point, dtype=torch.float64)
    
    # Initialize the Hypervolume calculator
    hypervolume_calculator = Hypervolume(ref_point=reference_point_tensor)
    
    # Store hypervolumes for each subset of points
    hypervolumes = []
    
    # Iterate through the list of points
    for i in range(1, len(points) + 1):
        current_subset = points_tensor[:i]
        hv = hypervolume_calculator.compute(pareto_Y=current_subset)
        hypervolumes.append(hv)
        print(f"Subset: {current_subset}, Hypervolume: {hv}")
    
    return hypervolumes

points = [
    (-542, 141.54),
    (-922, 159.60),
    (-423, 141.00),
    (-398, 135.53),
    (-764, 154.53),
    (-595, 138.12),
    (-700, 133.02),
    (-653, 155.03),
    (-738, 156.64),
    (-579, 152.60),
    (-504, 140.09),
    (-741, 158.54),
    (-820, 156.29),
    (-621, 146.39),
    (-404, 137.43),
    (-697, 154.55),
    (-792, 156.31),
    (-778, 159.37),
    (-541, 139.50),
    (-607, 153.23),
    (-680, 156.01),
    (-634, 157.92),
    (-708, 158.99),
    (-445, 140.53),
    (-655, 157.01),
    (-561, 151.14),
    (-771, 155.37),
    (-590, 145.61),
    (-527, 139.12),
    (-865, 151.64)
]
reference_point = (-1500.00, 100.00)

hypervolumes = calculate_hypervolume_for_points(points, reference_point)

print("Hypervolume values for each subset of points:")
for i, hv in enumerate(hypervolumes, start=1):
    print(f"{hv}")
