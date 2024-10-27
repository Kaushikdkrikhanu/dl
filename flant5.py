from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline
import torch
import json
import re
import os
import gc

# Load tokenizer and model for flan-t5-small
checkpoint = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float16)

# Set device to GPU (CUDA) if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the prompt as a single string for T5 input
zero_shot_prompt = "You are a math expert. Respond with the Solution, thinking step by step. Final Answer: The final answer is $answer$. I hope it is correct."

problem_prompt = "Simplify the following expression:\n1/5 ⋅ 2/7 ÷ 12/20."

# Define function to run the prompt through the model
def run(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_new_tokens=8192)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)
    return text

# Example usage of answer extraction function
def extract_answer(text):
    pattern = r'boxed{(.*?)}'
    match = re.search(pattern, text)
    return match.group(1) if match else None

# Define path and output file
directory_path = "MATH/train/algebra"
output_file = "answers.json"

# Initialize counters and answer lists
oracle_answers = []
first_answers = []
second_answers = []
correct1 = 0

# Function to save answers to JSON file
def save_answers_to_file(oracle_answers, first_answers, second_answers, output_file):
    data = {
        "oracle_answers": oracle_answers,
        "first_answers": first_answers,
        "second_answers": second_answers
    }
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Iterate over JSON files in directory
for filename in os.listdir(directory_path):
    gc.collect()
    torch.cuda.empty_cache()
    
    if filename.endswith(".json"):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            problem = json_data['problem']
            oracle_answer = json_data['solution']
            oracle_single_answer = extract_answer(oracle_answer)
            
            # Generate the model's answer
            model_answer = run(f"{zero_shot_prompt} {problem}")
            model_single_answer = extract_answer(model_answer)
            
            # Append the answers
            first_answers.append(model_answer)
            oracle_answers.append(oracle_answer)
            
            if oracle_single_answer == model_single_answer:
                correct1 += 1

            print(f"Accuracy1: {correct1 / len(first_answers):.2f}")

            # Self-correction
            self_correction_text = ("There might be an error. Correct and rewrite the solution as: "
                                    "Final Answer: The final answer is $answer$. I hope it is correct.")
            model_self_corrected_answer = run(f"{zero_shot_prompt} {problem} {model_answer} {self_correction_text}")
            second_answers.append(model_self_corrected_answer)

            # Save answers to JSON file after each file
            save_answers_to_file(oracle_answers, first_answers, second_answers, output_file)
