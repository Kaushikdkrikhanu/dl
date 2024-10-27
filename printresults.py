import numpy as np
import json
# Your provided data

import re

def extract_answer(text):
    """
    Extracts the content inside 'boxed{...}' from a given string.
    
    Args:
    text (str): The input string containing 'boxed{...}'.
    
    Returns:
    str: The extracted content inside 'boxed{...}', or None if no match is found.
    """
    # Regular expression to match the content inside 'boxed{...}'
    pattern = r'boxed{(.*?)}'

    # Use re.search to find the pattern
    matches = re.findall(pattern, text)

    # Extract the answer if a match is found
    if matches:
        return matches[-1]
    else:
        return None



with open("answers.json", 'r') as f:
    saved_data = json.load(f)
    oracle_answers = saved_data['oracle_answers']
    print("Total questions: ", len(oracle_answers))
    oracle_answers = [extract_answer(a) for a in oracle_answers]
    first_answers = saved_data['first_answers']
    first_answers = [extract_answer(a) for a in first_answers]
    second_answers = saved_data['second_answers'] 
    second_answers = [extract_answer(a) for a in second_answers]


# Function to compare answers, handling tuples and other data types
def compare_answers(answer1, answer2):
    return answer1 == answer2

# Convert the lists into numpy arrays of booleans, checking for equality
acc_t1_matches = np.array([compare_answers(o, f) for o, f in zip(oracle_answers, first_answers)])
acc_t2_matches = np.array([compare_answers(o, s) for o, s in zip(oracle_answers, second_answers)])

# Accuracy at t1 and t2
acc_t1 = np.mean(acc_t1_matches)
acc_t2 = np.mean(acc_t2_matches)

# Delta accuracy from t1 to t2
delta_acc = acc_t2 - acc_t1

# False to True (incorrect at t1, but correct at t2)
false_to_true = np.mean((~acc_t1_matches) & acc_t2_matches)

# True to False (correct at t1, but incorrect at t2)
true_to_false = np.mean(acc_t1_matches & (~acc_t2_matches))

# Display the results
print(f"Accuracy at t1: {acc_t1 * 100:.2f}%")
print(f"Accuracy at t2: {acc_t2 * 100:.2f}%")
print(f"Delta Accuracy (t1 to t2): {delta_acc * 100:.2f}%")
print(f"False to True ratio: {false_to_true * 100:.2f}%")
print(f"True to False ratio: {true_to_false * 100:.2f}%")
