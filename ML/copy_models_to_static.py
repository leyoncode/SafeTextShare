#!/usr/bin/env python3

import os
import sys

# File names
VECTORIZER_SAVE_PATH = 'vectorizer.pkl'
SAVED_MODEL_FILE_NAME = 'saved_model.pkl'

# Check for the -y flag in script arguments
auto_confirm = '-y' in sys.argv

print("Starting the file copy process...")

# Get current working directory
current_directory = os.getcwd()  # e.g., django_project/ML
print(f"Current working directory: {current_directory}")

# Get parent directory
parent_directory = os.path.dirname(current_directory)  # Parent directory, e.g., django_project
print(f"Parent directory: {parent_directory}")

# Define destination directory path
destination_directory = os.path.join(parent_directory, 'static')
print(f"Destination directory: {destination_directory}")

# Ensure the destination directory exists
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory, exist_ok=True)
    print(f"Destination directory created: {destination_directory}")
else:
    print(f"Destination directory already exists: {destination_directory}")

# Define source and destination paths for each file
files_to_copy = [VECTORIZER_SAVE_PATH, SAVED_MODEL_FILE_NAME]
print(f"Files to copy: {files_to_copy}")

for file_name in files_to_copy:
    source_path = os.path.join(current_directory, file_name)
    destination_path = os.path.join(destination_directory, file_name)

    print(f"Processing file: {file_name}")
    print(f"Source path: {source_path}")
    print(f"Destination path: {destination_path}")

    # Check if the file already exists in the destination
    if os.path.exists(destination_path):
        if not auto_confirm:
            # Ask for confirmation to overwrite the file, with default 'y'
            confirm = input(
                f"File {file_name} already exists at the destination. Do you want to overwrite it? (Y/n): ").strip().lower()
            if confirm not in ['', 'y']:
                print(f"Skipping {file_name} to avoid overwriting.")
                continue
        else:
            print(f"Auto-confirmation enabled. Overwriting {file_name}...")

    # Copy the file content
    try:
        with open(source_path, 'rb') as source_file:
            content = source_file.read()
        print(f"Read content from {source_path} successfully.")

        with open(destination_path, 'wb') as destination_file:
            destination_file.write(content)
        print(f"File {file_name} has been copied to {destination_path}")

    except FileNotFoundError:
        print(f"File not found: {source_path}")
    except Exception as e:
        print(f"An error occurred while copying {file_name}: {e}")

print("File copy process completed.")