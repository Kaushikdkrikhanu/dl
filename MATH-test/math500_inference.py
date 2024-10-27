from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_community.llms import Ollama


# Define the checkpoint for the Mathstral model
checkpoint = "meta-llama/Llama-3.2-1B-Instruct"

# Set device to GPU (CUDA) if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model with proper settings for device and dtype
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)  # Explicitly move to GPU

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

# Combine prompts into a single input for the model
def run(system_prompt, user_prompt):
    # Combine all messages into a single conversation
    full_prompt = ""
    for message in system_prompt + user_prompt:
        if message["role"] == "system":
            full_prompt += f"System: {message['content']}\n"
        elif message["role"] == "user":
            full_prompt += f"User: {message['content']}\n"
        elif message["role"] == "assistant":
            full_prompt += f"Assistant: {message['content']}\n"

    # Initialize Ollama
    llm = Ollama(model="llama3.2:1b")  # or whatever model you have pulled in Ollama
    
    # Get response from Ollama
    response = llm.invoke(full_prompt)
    
    return response


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
directory_path = "../sample"

# File where answers will be stored
output_file = "answer_pairs.json" 

# Initialize counters and answer lists
count = 0
correct1 = 0
correct2 = 0


def print_memory_stats():
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert to GB
    free_memory = reserved_memory - allocated_memory

    print(f"Allocated Memory: {allocated_memory:.2f} GB")
    print(f"Reserved Memory: {reserved_memory:.2f} GB")
    print(f"Free Memory: {free_memory:.2f} GB")
# Function to save lists to a JSON file
def save_answers_to_file_unique(oracle_answers, first_answers, second_answers, output_file):
    
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
    else:
        data = {
            "oracle_answers": [],
            "first_answers": [],
            "second_answers": []
        }
    
    # Append new data to existing lists
    data["oracle_answers"].extend([oracle_answers])
    data["first_answers"].extend([first_answers])
    data["second_answers"].extend([second_answers])
    
    # Write the updated data back to the file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def save_answers_to_file(oracle_answer, first_answer, second_answer, output_file):
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
    else:
        data = {
            "answer_pairs": []
        }
    
    # Extract single answers for comparison
    oracle_single = extract_answer(oracle_answer)
    first_single = extract_answer(first_answer)
    second_single = extract_answer(second_answer)
    
    # Create a new answer pair object with correctness flags
    answer_pair = {
        "oracle_answer": oracle_answer,
        "first_answer": first_answer,
        "second_answer": second_answer,
        "first_answer_correct": first_single == oracle_single if oracle_single and first_single else False,
        "second_answer_correct": second_single == oracle_single if oracle_single and second_single else False
    }
    
    # Append the new answer pair to the list
    data["answer_pairs"].append(answer_pair)
    
    # Write the updated data back to the file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Iterate over all files in the directory
with open('test.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if not line.strip():  # Skip empty lines
            continue
            
        print(i)
        
        # Parse the JSON line
        json_data = json.loads(line)
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
        first_answer = model_answer

        count += 1
        if oracle_single_answer == model_single_answer:
            correct1 += 1

        print(f"Accuracy1: {correct1/count:.2f}")

        # Self-correction process
        self_correction_text = 'There might be an error in the solution above...'

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

        second_answer = model_self_corrected_answer

        # Save the answers to the file after processing each JSON file
        save_answers_to_file(oracle_answer, first_answer, second_answer, output_file)
            

