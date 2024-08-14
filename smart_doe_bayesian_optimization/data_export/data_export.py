import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from models.gp_model import BaseModel
import torch
import pickle

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

    setup_information = {
    'Num Objectives': multiobjective_model.dataset_manager.output_dim,
    'Num Inputs': multiobjective_model.dataset_manager.input_dim,
    'Num Data Points': multiobjective_model.dataset_manager.initial_dataset.input_data.shape[0],
    'Variation Factor': multiobjective_model.dataset_manager.variation_factor,
    'Model Info (state dict)': multiobjective_model.gp_model.state_dict,
    'Model Hyperparameter': multiobjective_model.gp_model.hyperparameters,
    'Num historic datasets': len(multiobjective_model.dataset_manager.historic_modelinfo_list),
    'maximization_flags': multiobjective_model.dataset_manager.maximization_flags,  
    'bounds': multiobjective_model.dataset_manager.initial_dataset.bounds_list
    }   
    
    # Convert dictionaries to DataFrames
    optimization_df = pd.DataFrame.from_dict(optimization_dict, orient='index')
    pareto_points_df = pd.DataFrame(results_dict['pareto_points'])
    setup_information_df = pd.DataFrame.from_dict(setup_information, orient='index', columns=['Value'])

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
        setup_information_df.to_excel(writer, sheet_name='Setup Information')

    print(f"Data successfully saved to {file_path}")
    
    # Save each figure in the fig_list as a PNG
    for i, fig in enumerate(fig_list):
        fig_path = os.path.join(full_folder_path, f"figure_{i+1}.png")
        fig.savefig(fig_path)
        plt.close(fig)  # Close the figure to free memory
        #print(f"Figure {i+1} saved to {fig_path}")

    # Save the state_dict of the model
    state_dict_path = os.path.join(full_folder_path, f"{folder_name}_model_state_dict.pth")
    torch.save(multiobjective_model.gp_model.state_dict(), state_dict_path)
    model_info_dict = multiobjective_model.gp_model.state_dict()

    # save dataset infromation as pickle

    dataset_info_dict = {
        'num_inputs': multiobjective_model.dataset_manager.input_dim,
        'num_outputs': multiobjective_model.dataset_manager.output_dim,
        'num_datapoints': multiobjective_model.dataset_manager.initial_dataset.input_data.shape[0],
        'maximization_flags': multiobjective_model.dataset_manager.maximization_flags,
        'bounds': multiobjective_model.dataset_manager.initial_dataset.bounds_list,
        'variation_factor': multiobjective_model.dataset_manager.variation_factor
        #here potential additional information can be added
    }

    # Save initial dataset as pickle
    dataset_dict = {
        'identifier': multiobjective_model.dataset_manager.initial_dataset.identifier,
        'input_data': input_data,
        'output_data': output_data,
        'bounds': multiobjective_model.dataset_manager.initial_dataset.bounds_list,
        'maximization_flags': multiobjective_model.dataset_manager.initial_dataset.maximization_flags,
    }

    pickle_path = os.path.join(full_folder_path, f"{folder_name}_dataset.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(dataset_dict, f)

    #save all three dicts together in one pkl:
    combined_dict = {
        'model_info': model_info_dict,
        'dataset_info': dataset_info_dict,
        'dataset': dataset_dict
    }

    combined_pickle_path = os.path.join(full_folder_path, f"{folder_name}_combined_dataset_info.pkl")

    with open(combined_pickle_path, 'wb') as f:
        pickle.dump(combined_dict, f)

    print(f"All figures saved to {full_folder_path}")

def export_only_in_out_data(input_data: torch.Tensor, output_data: torch.Tensor, folder_path: str, folder_name: str):

    # Create the folder if it doesn't exist
    full_folder_path = os.path.join(folder_path, folder_name)
    os.makedirs(full_folder_path, exist_ok=True)
    
    # Define the file path for the Excel file
    file_path = os.path.join(full_folder_path, f"{folder_name}_results.xlsx")

    input_data = input_data.numpy() #shape: Tensor of shape (n, d)
    output_data = output_data.numpy() #shape: Tensor of shape (n, d)

    input_df = pd.DataFrame(input_data)
    output_df = pd.DataFrame(output_data)

    # Save the DataFrames to Excel
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        input_df.to_excel(writer, sheet_name='Input Data')
        output_df.to_excel(writer, sheet_name='Output Data')