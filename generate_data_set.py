import os
import json

def extract_level_1_problems(input_directory, output_file):
    # Create a list to store all Level 1 problems
    level_1_problems = []

    # Walk through all files and folders in the input directory
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".json"):  # Process only JSON files
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Check if it's a Level 1 problem
                        if data.get("level") == "Level 1":
                            level_1_problems.append(data)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    # Write all Level 1 problems to the output JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(level_1_problems, f, indent=4)
        print(f"Extracted {len(level_1_problems)} Level 1 problems into {output_file}.")
    except Exception as e:
        print(f"Error writing to output file {output_file}: {e}")


# Define input folder and output JSON file
input_directory = "MATH/train"  # Replace with the path to your folder
output_file = "all_level_1_problems.json"   # Replace with the desired output file name

# Run the extraction
extract_level_1_problems(input_directory, output_file)
