import argparse
import json
import os
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_json_files(directory_path: str):
    """
    Recursively finds all JSON files in the given directory.

    Args:
        directory_path (str): The path to the directory to search.

    Yields:
        str: The full path to each JSON file found.
    """
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".json"):
                yield os.path.join(root, file)

def extract_costs_from_files(json_file_paths):
    """
    Extracts 'estimated_cost_usd' from a list of JSON files.

    Args:
        json_file_paths (iterable): An iterable of paths to JSON files.

    Returns:
        list[float]: A list of extracted costs.
    """
    costs = []
    for file_path in json_file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            cost = data.get('estimated_cost_usd')
            if cost is not None:
                if isinstance(cost, (int, float)):
                    costs.append(float(cost))
                else:
                    logger.warning(f"Cost value in {file_path} is not a number: {cost}")
            else:
                logger.warning(f"Key 'estimated_cost_usd' not found in {file_path}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from file: {file_path}")
        except Exception as e:
            logger.error(f"An unexpected error occurred with file {file_path}: {e}")
    return costs

def main():
    """
    Main function to parse arguments, find JSON files, extract costs,
    and calculate statistics.
    """
    parser = argparse.ArgumentParser(
        description="Calculate statistics for 'estimated_cost_usd' from JSON files in a directory."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing JSON files to process.",
    )
    args = parser.parse_args()

    logger.info(f"Scanning directory: {args.input_dir}")
    
    json_files = list(find_json_files(args.input_dir))
    
    if not json_files:
        logger.info("No JSON files found in the specified directory.")
        return

    logger.info(f"Found {len(json_files)} JSON files. Extracting costs...")
    
    costs = extract_costs_from_files(json_files)

    if not costs:
        logger.info("No 'estimated_cost_usd' values found in the JSON files or all values were invalid.")
        return

    logger.info(f"Successfully extracted {len(costs)} cost values.")

    # Calculate statistics
    total_cost = np.sum(costs)
    mean_cost = np.mean(costs)
    std_dev_cost = np.std(costs)
    min_cost = np.min(costs)
    max_cost = np.max(costs)

    print("\n--- Cost Statistics ---")
    print(f"Total Estimated Cost (USD): {total_cost:.4f}")
    print(f"Mean Estimated Cost (USD):  {mean_cost:.4f}")
    print(f"Std Dev Estimated Cost (USD): {std_dev_cost:.4f}")
    print(f"Min Estimated Cost (USD):   {min_cost:.4f}")
    print(f"Max Estimated Cost (USD):   {max_cost:.4f}")
    print(f"Number of data points:      {len(costs)}")
    print("-------------------------")

if __name__ == "__main__":
    main() 