import os
import argparse

def remove_incremental_files(parent_folder_path):
    # Check if the provided path is a valid directory
    if not os.path.isdir(parent_folder_path):
        print(f"The provided path '{parent_folder_path}' is not a valid directory.")
        return

    # Walk through the directory tree
    for root, dirs, files in os.walk(parent_folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            
            # Check if the file path contains the string 'incremental'
            if '/incremental/model_ep' in file_path:
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Remove files containing 'incremental' in their names from the specified folder and its subfolders.")
    parser.add_argument('parent_folder_path', type=str, help="Path to the parent folder where files will be removed.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function with the provided parent folder path
    remove_incremental_files(args.parent_folder_path)
