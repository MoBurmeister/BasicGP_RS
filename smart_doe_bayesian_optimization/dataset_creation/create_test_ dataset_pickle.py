import torch
import pickle
import os

def create_dataset(identifier, n, d):
    return {
        'identifier': identifier,
        'input_dataset': torch.randn(n, d),
        'output_dataset': torch.randn(n, 3),
        'bounds': torch.tensor([[0.0] * d, [1.0] * d])
    }

def main():
    num_datasets = 5  # Number of datasets to create
    n = 10  # Number of rows in each dataset
    d = 5   # Number of columns in each dataset

    datasets_list = [create_dataset(i, n, d) for i in range(1, num_datasets + 1)]

    # Define the path to save the pickle file
    save_path = os.path.join('smart_doe_bayesian_optimization', 'dataset_creation', 'pickle_files')
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, 'datasets.pkl')

    # Save the list of datasets to a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(datasets_list, f)

    print(f"Datasets saved to {file_path}")

if __name__ == "__main__":
    main()

