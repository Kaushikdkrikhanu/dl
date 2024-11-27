import json
from collections import defaultdict
import random

def select_problems_by_type(input_file, output_file, count_per_type=150):
    """
    Selects a specified number of problems for each type and saves them to a new JSON file.

    Parameters:
        input_file (str): Path to the input JSON file containing all problems.
        output_file (str): Path to the output JSON file to save selected problems.
        count_per_type (int): Number of problems to select per type.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data['answer_pairs']
            # Ensure the data is a list
            if not isinstance(data, list):
                print("The input file does not contain a list of problems.")
                return
            
            # Group problems by type
            problems= []
            for problem in data:
                if problem['first_answer_correct']:
                    problems.append(problem)
            
            # Select `count_per_type` problems from each type
            # selected_problems = []
            # for problem_type, problems in problems_by_type.items():
            #     if len(problems) <= count_per_type:
            #         selected_problems.extend(problems)  # Take all if less than required
            #     else:
            #         selected_problems.extend(random.sample(problems, count_per_type))  # Randomly select required count
            
            # Save selected problems to the output JSON file
            with open(output_file, 'w', encoding='utf-8') as f_out:
                json.dump(problems, f_out, indent=4, ensure_ascii=False)
            
            print(f"Selected {count_per_type} problems from each type and saved to {output_file}.")
    
    except FileNotFoundError:
        print(f"File not found: {input_file}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

# Usage
input_json = "1b_original_answer.json"  # Replace with the path to your input JSON file
output_json = "selected_problems_105_Level1.json"  # Replace with the path to your output JSON file
select_problems_by_type(input_json, output_json, count_per_type=15)
