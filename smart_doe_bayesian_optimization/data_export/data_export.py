import pandas as pd
import numpy as np

def export_dicts(optimization_dict: dict, results_dict: dict, file_path: str, file_format: str = "xlsx"):
    """
    Exports the optimization_dict and results_dict to a file with multiple sheets.

    Args:
        optimization_dict (dict): The optimization_dict to export.
        results_dict (dict): The results_dict to export.
        file_path (str): The path to the file.
        file_format (str, optional): The format of the file. Defaults to "xlsx".
    """
    if file_format != "xlsx":
        raise ValueError("Currently, only XLSX format is supported.")
    
    print(optimization_dict)
    print(results_dict)

    # Convert the dictionaries to pandas DataFrames
    optimization_df = pd.DataFrame.from_dict(optimization_dict, orient='index')

    pareto_points_df = pd.DataFrame(results_dict['pareto_points'].numpy())
    
    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        # Write each DataFrame to a different worksheet
        optimization_df.to_excel(writer, sheet_name='Optimization')
        pareto_points_df.to_excel(writer, sheet_name='Results')