from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define the checkpoint for the Mathstral model
checkpoint = "mistralai/Mathstral-7b-v0.1"

# Set device to GPU (CUDA) if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model with proper settings for device and dtype
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)  # Explicitly move to GPU

zero_shot_prompt = [
    {
        "role": "system",
        "content": (
            "You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking "
            "step by step. At the end of the Solution, when you give your final answer, write it in the form "
            "'Final Answer: The final answer is $answer$. I hope it is correct.'"
        )
    }
]

problem_prompt = [
    {
        "role": "user",
        "content": (
            "Simplify the following expression:\n"
            "1/5 ⋅ 2/7 ÷ 12/20."
        )
    }
]


def print_memory_stats():
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert to GB
    free_memory = reserved_memory - allocated_memory

    print(f"Allocated Memory: {allocated_memory:.2f} GB")
    print(f"Reserved Memory: {reserved_memory:.2f} GB")
    print(f"Free Memory: {free_memory:.2f} GB")

# Combine prompts into a single input for the model
def run(system_prompt, user_prompt): 
    full_prompt = system_prompt + user_prompt

    # Tokenize the combined prompt using the chat template and move it to the GPU
    tokenized_prompt = tokenizer.apply_chat_template(full_prompt, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)

    # Use no_grad to prevent memory accumulation from gradient tracking
    with torch.no_grad():
        # Generate output from the model
        out = model.generate(
            **tokenized_prompt,
            max_new_tokens=2048
        )

    # Decode and print the generated response
    generated_text = tokenizer.decode(out[0], skip_special_tokens=True)
 
    #attempt to clear memory
    del tokenized_prompt, out
    gc.collect()
    torch.cuda.empty_cache()

    print_memory_stats()
    return generated_text


# !tar -xvf MATH.tar

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
    match = re.search(pattern, text)

    # Extract the answer if a match is found
    if match:
        return match.group(1)  # The first capturing group
    else:
        return None

# Example usage:
text = "adasbda boxed{answer} asdasda"
answer = extract_answer(text)
if answer:
    print(f"Extracted answer: {answer}")
else:
    print("No match found")


import os
import json
import gc
import torch

gc.collect()  # These commands help you when you face CUDA OOM error
torch.cuda.empty_cache()

# Define the path to the directory containing JSON files
directory_path = "MATH/train/algebra"

# File where answers will be stored
output_file = "answers.json"

# Initialize counters and answer lists
count = 0
oracle_answers = []
first_answers = []
second_answers = []
correct1 = 0
correct2 = 0

# Function to save lists to a JSON file
def save_answers_to_file(oracle_answers, first_answers, second_answers, output_file):
    data = {
        "oracle_answers": oracle_answers,
        "first_answers": first_answers,
        "second_answers": second_answers
    }
    
    # Open the file in write mode to overwrite with new content
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    # gc.collect()  # These commands help you when you face CUDA OOM error
    # torch.cuda.empty_cache()
    
    if filename.endswith(".json"):  # Check if the file is a JSON file
        file_path = os.path.join(directory_path, filename)
        
        # Open and load the JSON file into a dictionary
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            problem_prompt = [
                {
                    "role": "user",
                    "content": (json_data['problem'])
                }
            ]
            
            oracle_answer = json_data['solution']
            oracle_single_answer = extract_answer(oracle_answer)

            model_answer = run(zero_shot_prompt, problem_prompt)
            model_single_answer = extract_answer(model_answer)
            
            # Append the answers
            first_answers.append(model_answer)
            oracle_answers.append(oracle_answer)
            
            count += 1
            if oracle_single_answer == model_single_answer:
                correct1 += 1

            print(f"Accuracy1: {correct1/count:.2f}")

            # Self-correction process
            self_correction_text = 'There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."'

            model_answer_prompt = [
                {
                    "role": "assistant",
                    "content": (model_answer)
                }
            ]

            self_correction_prompt = [
                {
                    "role": "user",
                    "content": (self_correction_text)
                }
            ]

            model_self_corrected_answer = run(zero_shot_prompt+problem_prompt+model_answer_prompt, self_correction_prompt)

            second_answers.append(model_self_corrected_answer)

            # Save the answers to the file after processing each JSON file
            save_answers_to_file(oracle_answers, first_answers, second_answers, output_file)
            

