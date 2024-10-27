import json
import re

def has_boxed_syntax(text):
    """Check if the text contains \boxed{} syntax."""
    pattern = r'\\boxed{.*?}'
    return bool(re.search(pattern, text))

def analyze_answers(file_path):
    """Analyze answer pairs and return array of problem indices."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    missing_boxed_indices = []
    
    for idx, pair in enumerate(data['answer_pairs']):
        if not pair['first_answer_correct']:
            if has_boxed_syntax(pair['oracle_answer']) and not has_boxed_syntax(pair['first_answer']):
                missing_boxed_indices.append(idx + 1)  # Adding 1 to make it 1-based indexing
    
    return missing_boxed_indices

def main():
    input_file = "answer_pairs.json"
    
    try:
        problem_numbers = analyze_answers(input_file)
        print("Problems with missing boxed syntax:")
        print(problem_numbers)
        print(f"\nTotal problems found: {len(problem_numbers)}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the input file '{input_file}'")
    except json.JSONDecodeError:
        print(f"Error: The file '{input_file}' is not valid JSON")

if __name__ == "__main__":
    main()