import os
import json

def extract_level_1_problems(input_directory, output_file1, output_file2):
    # Create a list to store all Level 1 problems
    train_problems = []
    test_problems = []
    count = 100
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
                            if count > 0: 
                                count = count - 1
                                test_problems.append(data)
                            else:
                                train_problems.append(data)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    # Write all Level 1 problems to the output JSON file
    try:
        with open(output_file1, 'w') as f:
            json.dump(test_problems, f, indent=4)
            
        with open(output_file2, 'w') as f:
            json.dump(train_problems, f, indent=4)
        print(f"Extracted {len(test_problems)} Level 1 problems into {output_file1}.")
        print(f"Extracted {len(train_problems)} Level 1 problems into {output_file2}.")
        
    except Exception as e:
        print(f"Error writing to output file {e}")


# Define input folder and output JSON file
input_directory = "MATH/train"  # Replace with the path to your folder
output_file2 = "train_problems_level1.json"   # Replace with the desired output file name
output_file1 = "test_problems_level1.json"
# Run the extraction
extract_level_1_problems(input_directory, output_file1, output_file2)
