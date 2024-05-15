#utils to convert files/types/etc.
import matplotlib.pyplot as plt
import os

def matplotlib_to_png(plot, filename, path, format='png'):
    """
    Save a matplotlib plot as a PNG image.

    Parameters:
    - plot: The matplotlib plot object to be saved.
    - filename: The name of the output file (without extension).
    - path: The path where the output file will be saved.
    - format: The format of the output file (default is 'png').

    Returns:
    - None
    """
    print(f"PATH: {path}")
    
    full_path = os.path.join(path, filename)

    plot.savefig(full_path, format=format)
