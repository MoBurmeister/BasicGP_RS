import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from models.gp_model import BaseModel

def export_everything(multiobjective_model: BaseModel, optimization_dict: dict, results_dict: dict, fig_list,  folder_path: str, folder_name: str, file_format: str = "xlsx"):
    # Check for the correct file format
    if file_format != "xlsx":
        raise ValueError("Currently, only XLSX format is supported.")
    
    # Create the folder if it doesn't exist
    full_folder_path = os.path.join(folder_path, folder_name)
    os.makedirs(full_folder_path, exist_ok=True)
    
    # Define the file path for the Excel file
    file_path = os.path.join(full_folder_path, f"{folder_name}_results.{file_format}")
    
    # Print the dictionaries (for debugging purposes)
    #print("Optimization Dictionary:", optimization_dict)
    #print("Results Dictionary:", results_dict)
    
    # Convert dictionaries to DataFrames
    optimization_df = pd.DataFrame.from_dict(optimization_dict, orient='index')
    pareto_points_df = pd.DataFrame(results_dict['pareto_points'])

    input_data = multiobjective_model.dataset_manager.initial_dataset.input_data.numpy() #shape: Tensor of shape (n, d)
    output_data = multiobjective_model.dataset_manager.initial_dataset.output_data.numpy() #shape: Tensor of shape (n, d)

    input_df = pd.DataFrame(input_data)
    output_df = pd.DataFrame(output_data)
    
    # Save the DataFrames to Excel
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        optimization_df.to_excel(writer, sheet_name='Optimization')
        pareto_points_df.to_excel(writer, sheet_name='Results')
        input_df.to_excel(writer, sheet_name='Input Data')
        output_df.to_excel(writer, sheet_name='Output Data')

    print(f"Data successfully saved to {file_path}")
    
    # Save each figure in the fig_list as a PNG
    for i, fig in enumerate(fig_list):
        fig_path = os.path.join(full_folder_path, f"figure_{i+1}.png")
        fig.savefig(fig_path)
        plt.close(fig)  # Close the figure to free memory
        print(f"Figure {i+1} saved to {fig_path}")

    print(f"All figures saved to {full_folder_path}")