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
            "'Final Answer: The final answer is \\boxed{answer}. I hope it is correct.'"
        )
    }
]

# problem_prompt = [
#     {
#         "role": "user",
#         "content": (
#             "Simplify the following expression:\n"
#             "1/5 ⋅ 2/7 ÷ 12/20."
#         )
#     }
# ]

# review_prompt = f"""Review the following solution to this math problem:
#         Question: {question}
#         Solution: {current_solution}
        
#         Please identify any errors or areas for improvement. Be specific about:
#         1. Calculation errors
#         2. Logical flaws
#         3. Missing steps
#         4. Unclear explanations
        
#         Return your review as: correct (True/False) and feedback (specific issues found)"""
        
# self_refine_prompt = f"""
#                 Provide a solution to the problem considering the feedback.
#                 """

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
directory_path = "test.jsonl"

# File where answers will be stored
output_file = "self_refine_answer_pairs.json" 

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

def save_answers_to_file(oracle_answer, first_answer, second_answer, output_file, json_data):
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
    
    # Create a new answer pair object with all fields
    answer_pair = {
        # Original problem data
        "problem": json_data['problem'],
        "solution": json_data['solution'],
        "answer": json_data['answer'],
        "subject": json_data['subject'],
        "level": json_data['level'],
        "unique_id": json_data['unique_id'],
        
        # Model responses and evaluation
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

import time
# Iterate over all files in the directory
with open(directory_path, 'r') as f:
    for i, line in enumerate(f):
        if i<205:
            continue
        if not line.strip():  # Skip empty lines
            continue
        
        print(f"Processing entry {i}")
        
        start_time = time.time()
        json_data = json.loads(line)
        problem_prompt = [
            {
                "role": "user",
                "content": (json_data['problem'])
            }
        ]
        
        oracle_answer = json_data['solution']
        oracle_single_answer = extract_answer(oracle_answer)
    
        # model_answer = run(zero_shot_prompt, problem_prompt)
        # model_single_answer = extract_answer(model_answer)
        
        # Initial answer generation
        first_answer = run(zero_shot_prompt, problem_prompt)
        first_single_answer = extract_answer(first_answer)
        if oracle_single_answer == first_single_answer:
            correct1 += 1
        print(f"Initial accuracy: {correct1 / (i + 1):.2f}")

        # Feedback generation
        feedback_prompt = [
            {"role": "user", "content": problem_prompt},
            {"role": "assistant", "content": first_answer},
            {"role": "user", "content": "Provide feedback on the solution above in terms of calculation errors, logical flaws, or missing steps."}
        ]
        feedback = run(zero_shot_prompt, feedback_prompt)
        
        
        # Self-correction with feedback
        # correction_prompt = problem_prompt +  [{"role": "assistant", "content": feedback}]
        self_refine_prompt = [ {"role" : "user", 
                         "content" : "Given the feedback on your initial solution, please revise your answer to correct any errors or fill in any missing steps. "
                                     "Ensure that the revised answer is accurate and follows a clear step-by-step approach to arrive at the final solution. "
                                     "Provide your final answer in the form: 'Final Answer: The final answer is \\boxed{answer}.'"}]
        correction_prompt = feedback_prompt + [{"role": "assistant", "content": feedback}]  + self_refine_prompt
        second_answer = run(zero_shot_prompt, correction_prompt)
        second_single_answer = extract_answer(second_answer)
        if oracle_single_answer == second_single_answer:
            correct2 += 1
        print(f"Self-corrected accuracy: {correct2 / (i + 1):.2f}")
        
        
        # Append the answers
        # first_answer = model_answer

        # count += 1
        # if oracle_single_answer == model_single_answer:
        #     correct1 += 1

        # print(f"Accuracy1: {correct1/count:.2f}")

        # Self-correction process
        # self_correction_text = 'There might be an error in the solution above...'
        # model_answer_prompt = [
        #     {
        #             "role": "assistant",
        #             "content": (model_answer)
        #         }
        #     ]

        # self_correction_prompt = [
        #         {
        #             "role": "user",
        #             "content": (self_correction_text)
        #         }
        #     ]

        # model_self_corrected_answer = run(zero_shot_prompt+problem_prompt+model_answer_prompt, self_correction_prompt)
        # second_answer = model_self_corrected_answer

        # Save the answers to the file after processing each JSON file
        save_answers_to_file(oracle_answer, first_answer, second_answer, output_file, json_data)
        
        elapsed_time = time.time() - start_time
        if elapsed_time > 60:  # 60 seconds = 1 minute
            print(f"Warning: Inference {i} took {elapsed_time:.2f} seconds (more than 1 minute)")
            

