import gc
import os
import torch
from torch.utils.data import DataLoader
import helper_utils
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = Path.cwd() / "data/cifar10"
trainset = helper_utils.download_and_load_cifar10(data_path)

def custom_experiment(trainset, device):
    """
    Runs a custom experiment to measure DataLoader performance.

    Args:
        trainset: The dataset to be used for the experiment.
        device: The device (e.g., 'cpu' or 'cuda') on which to run the test.

    Returns:
        A tuple containing:
            - A dictionary with the performance results.
            - The name of the parameter that was tested.
    """
    
    # Specify the name of the DataLoader parameter to be tested.
    # For example: parameter_name = 'prefetch_factor'
    
    parameter_name = 'prefetch_factor'

    # Provide a list of values to iterate through for the specified parameter.
    # For example: list_of_values_to_test = [6, 8]
    
    list_of_values_to_test = [6, 8]

    # Initialize an empty dictionary to store the performance results.
    results_dictionary = {}

    # Iterate over each value in the test list.
    for current_value in list_of_values_to_test:
        print(f"--- Testing {parameter_name} = {current_value} ---")
        
        # Configure and instantiate the DataLoader for the current test iteration.
        # For example: loader = DataLoader(trainset, 
                                       #  batch_size=64, 
                                       #  shuffle=True, 
                                       #  num_workers=2, 
                                       #  pin_memory=False,
                                       #  prefetch_factor=current_value
                                       # )
        
        loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, pin_memory=False, prefetch_factor=current_value)
        
        # Measure the performance and handle potential runtime errors.
        try:
            # Calculate the average epoch time and store it in the results dictionary.
            results_dictionary[current_value] = helper_utils.measure_average_epoch_time(loader, device)
        except RuntimeError as e:
            # Handle cases where a runtime error occurs, such as an out-of-memory issue.
            print(f"\n❌ ERROR with {parameter_name} = {current_value}: {e}")
            results_dictionary[current_value] = float('inf')
            
        # Ensure each test run is independent by cleaning up memory.
        # Clean up the DataLoader instance to free up resources.
        del loader
        # Invoke the garbage collector to release unreferenced memory.
        gc.collect()

        # Clear the CUDA cache if a GPU is available.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Return the dictionary of results and the name of the tested parameter.
    return results_dictionary, parameter_name

results_dictionary, parameter_name = custom_experiment(trainset=trainset, device=device)

helper_utils.plot_performance_summary(
    results_dictionary, 
    title=f"DataLoader Performance vs. {parameter_name}", 
    xlabel=parameter_name.replace('_', ' ').title(), 
    ylabel="Average Time per Epoch (milliseconds)"
)
