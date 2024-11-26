import os
import random
import json
import threading
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing_extensions import TypedDict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    StoppingCriteria, StoppingCriteriaList
)
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import radon.complexity as radon_complexity
from sympy import simplify, SympifyError
from sympy.parsing.sympy_parser import parse_expr
import ast

# Initialize NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Configure logging
logging.basicConfig(
    level='INFO',
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # if torch.backends.mps.is_available():
        #     torch.mps.manual_seed(seed)
        logger.info(f"Seed set to {seed}.")
    except Exception as e:
        logger.error(f"Error setting seed: {e}")
        raise RuntimeError("Failed to set seed.") from e


@dataclass
class Config:
    """
    Configuration dataclass for training parameters.
    """
    beta_1: float = 0.01
    beta_2: float = 0.1
    alpha: float = 5.0
    learning_rate: float = 1e-5
    batch_size: int = 1
    max_seq_len: int = 4096
    max_new_tokens: int = 4096
    num_epochs_stage_one: int = 1
    num_epochs_stage_two: int = 1

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device: torch.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    seed: int = 42
    task: str = 'MATH'
    model_variant: str = 'meta-llama/Llama-3.2-1B-Instruct'
    ablation: str = 'none'
    data_path: str = './data'
    output_dir: str = './outputs'
    num_workers: int = 2
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    save_steps: int = 1000
    logging_steps: int = 10
    eval_steps: int = 1000
    max_eval_samples: int = 500
    mixed_precision: bool = False
    save_total_limit: int = 2
    compute_bleu: bool = False
    compute_rouge: bool = False
    compute_cyclomatic_complexity: bool = False

    def validate(self) -> None:
        """
        Validate configuration parameters.
        """
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer.")
        if self.num_epochs_stage_one < 0 or self.num_epochs_stage_two < 0:
            raise ValueError("Number of epochs must be non-negative.")
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        if not os.path.isdir(self.output_dir):
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                logger.info(f"Created output directory at {self.output_dir}.")
            except Exception as e:
                logger.error(f"Failed to create output directory: {e}")
                raise


def get_math_first_turn_prompt(problem: str) -> str:
    """Generate the first turn prompt for math problems."""
    return (
        "<|system|>You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking "
        "step by step. At the end of the Solution, when you give your final answer, write it in the form "
        "'Final Answer: The final answer is \\boxed{answer}. I hope it is correct.'"
        "</|system|>\n"
        f"<|user|>{problem}</|user|>\n"
        "<|assistant|>" 
    )

def get_math_correction_prompt(problem: str, prev_attempt: str) -> str:
    """Generate the self-correction prompt for math problems."""
    return (
        "<|system|>There might be an error in the solution above because of lack of understanding of the question. "
        "Please correct the error, if any, and rewrite the solution. Only output the final solution! "
        "At the end of the Solution, when you give your final answer, write it in the form "
        "'Final Answer: The final answer is \\boxed{answer}. I hope it is correct.'"
        "</|system|>\n"
        f"<|user|>Problem: {problem}\n\n"
        f"Previous solution:\n{prev_attempt}\n"
        "Please provide a corrected solution.</|user|>\n"
        "<|assistant|>"
    )


def get_code_first_turn_prompt(problem: str) -> str:
    """Generate the first turn prompt for code problems.
    
    Args:
        problem (str): Problem description including function signature and test cases
        
    Returns:
        str: Formatted prompt for first attempt
    """
    return (
        "<|system|>You are an expert Python programmer, and here is your task: Write a function to find the similar elements from"
        "the given two tuple lists. Your code should pass these tests:</|system|>\n"
        f"<|user|>{problem}</|user|>\n"
        "<|assistant|>"
    )

def get_code_correction_prompt(problem: str, prev_attempt: str) -> str:
    """Generate the self-correction prompt for code problems.
    
    Args:
        problem (str): Original problem description including function signature and test cases 
        prev_attempt (str): Previous code attempt to be corrected
        
    Returns:
        str: Formatted prompt for correction attempt
    """
    return (
        "<|system|>There might be an error in the code above because of lack of understanding of the question. Please correct the "
        "error, if any, and rewrite the solution. Only output the final correct Python program!</|system|>\n"
        f"<|user|>Problem:\n{problem}\n\n"
        f"Previous solution:\n{prev_attempt}\n\n"
        "Please provide a corrected solution.</|user|>\n"
        "<|assistant|>"
    )


class BaseDataset(Dataset):
    """
    Base dataset class for loading data.
    """

    def __init__(self, data: List[Dict[str, Any]], task: str = 'MATH'):
        self.data = data
        self.task = task

    def __len__(self) -> int:
        return len(self.data)
    def prepare_prompt(self, item: Dict[str, Any], turn: int = 1, prev_attempt: Optional[str] = None) -> str:
        """
        Prepare prompt based on task and turn number.
        
        Args:
            item: Data item containing problem/prompt
            turn: Turn number (1 or 2)
            prev_attempt: Previous attempt for turn 2
            
        Returns:
            Formatted prompt string
        """
        if self.task == 'MATH':
            if turn == 1:
                return get_math_first_turn_prompt(item['problem'])
            else:
                return get_math_correction_prompt(item['problem'], prev_attempt)
        else:
            # Handle other tasks like CODE similarly
            raise NotImplementedError("Only MATH task is implemented")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            item = self.data[idx]
            # Format prompt for first turn
            item['formatted_prompt'] = self.prepare_prompt(item)
            return item
        except IndexError as e:
            logger.error(f"Index {idx} out of range for dataset of size {len(self.data)}.")
            raise IndexError("Dataset index out of range.") from e
        except Exception as e:
            logger.error(f"Error retrieving item at index {idx}: {e}")
            raise


def load_json(file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load data from a JSON or JSONL file.

    Args:
        file_path (str): Path to the JSON or JSONL file.
        max_samples (Optional[int]): Maximum number of samples to load.

    Returns:
        List[Dict[str, Any]]: Loaded data.
    """
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be a non-negative integer or None")

    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                for idx, line in enumerate(f):
                    if max_samples is not None and idx >= max_samples:
                        break
                    if line.strip():  # Skip empty lines
                        data.append(json.loads(line))
            else:
                file_content = f.read().strip()
                if file_content:
                    loaded_data = json.loads(file_content)
                    if isinstance(loaded_data, list):
                        data = loaded_data[:max_samples] if max_samples else loaded_data
                    else:
                        data = [loaded_data]
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}") from e
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in file {file_path}: {e}")
        raise ValueError(f"Invalid JSON format in file: {file_path}") from e
    except Exception as e:
        logger.error(f"Unexpected error while loading JSON from {file_path}: {e}")
        raise RuntimeError(f"Failed to load data from {file_path}") from e

    logger.info(f"Loaded {len(data)} samples from {file_path}.")
    return data


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for text generation."""
    def __init__(self, stop_token_ids: List[List[int]], min_length: int = 20):
        self.stop_token_ids = stop_token_ids
        self.min_length = min_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Don't stop if we haven't generated minimum length
        if input_ids.shape[-1] < self.min_length:
            return False
            
        # Check for stop sequences
        for stop_ids in self.stop_token_ids:
            if len(stop_ids) > 0 and torch.all((input_ids[0][-len(stop_ids):] == torch.tensor(stop_ids).to(input_ids.device))).item():
                return True
        return False

class AdvancedModel(nn.Module):
    """
    Advanced model wrapper with tokenizer and generation capabilities.
    """

    def __init__(self, model_name: str, device: torch.device):
        super().__init__()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                padding_side='left'
            )
            self.system_marker = "<|system|>"
            self.user_marker = "<|user|>"
            self.assistant_marker = "<|assistant|>"
            self.stop_sequences = [
                self.system_marker,
                self.user_marker,
                "Previous Attempt:",
                "Instructions:"
            ]
            logger.info(f"Tokenizer loaded for {model_name}.")
        except Exception as e:
            logger.error(f"Error loading tokenizer for {model_name}: {e}")
            raise RuntimeError(f"Failed to load tokenizer for {model_name}") from e

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Using EOS token as PAD token.")

        try:
            self.model = LlamaForCausalLM.from_pretrained(model_name).to(device)
            logger.info(f"Model loaded and moved to {device}.")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise RuntimeError(f"Failed to load model {model_name}") from e

        try:
            if not self.tokenizer.pad_token:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info("Added pad token and resized token embeddings.")
        except Exception as e:
            logger.error(f"Error adding pad token or resizing embeddings: {e}")
            raise RuntimeError("Failed to add pad token or resize embeddings.") from e

        self.device = device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Logits from the model.
        """
        try:
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            raise RuntimeError("Forward pass failed.") from e

    def clean_response(self, text: str) -> str:
        """Clean the generated response by removing prompt repetition and extracting only the solution."""
        try:
            # Split on assistant marker and take the last part
            if self.assistant_marker in text:
                text = text.split(self.assistant_marker)[-1]
            
            # Remove any remaining system/user markers and their content
            if self.system_marker in text:
                text = text.split(self.system_marker)[0]
            if self.user_marker in text:
                text = text.split(self.user_marker)[0]
            
            # Remove common prefixes and their variations
            prefixes_to_remove = [
                "Step-by-step solution:",
                "Step by step solution:",
                "Previous Attempt:",
                "Instructions:",
                "Here's the solution:",
                "Solution:",
                "Let me solve this",
                "Please provide a corrected solution."
            ]
            
            # Clean up the text
            text = text.strip()
            for prefix in prefixes_to_remove:
                if text.lower().startswith(prefix.lower()):
                    text = text[len(prefix):].strip()
            
            # Clean up any remaining markers
            text = text.replace(self.system_marker, "")
            text = text.replace(self.user_marker, "")
            text = text.replace(self.assistant_marker, "")
            
            # Remove multiple newlines and clean up spacing
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)
            
            # If the text still starts with common prefixes after newlines, remove them
            for prefix in prefixes_to_remove:
                if text.lower().startswith(prefix.lower()):
                    text = text[len(prefix):].strip()
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error cleaning response: {e}")
            return text.strip()
        
    def generate_text(
        self,
        inputs: Dict[str, torch.Tensor],
        max_length: int = 4096,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        min_length: int = 20  # Minimum length of generated text
    ) -> torch.Tensor:
        """
        Generate text using the model.

        Args:
            inputs (Dict[str, torch.Tensor]): Tokenized inputs.
            max_length (int): Maximum length of generated text.
            temperature (float): Sampling temperature.
            num_return_sequences (int): Number of sequences to generate.

        Returns:
            torch.Tensor: Generated token IDs.
        """
        try:
            stop_token_ids = [
                self.tokenizer.encode(seq, add_special_tokens=False)
                for seq in self.stop_sequences
            ]

            # Create stopping criteria with minimum length
            stopping_criteria = StoppingCriteriaList([
                StopOnTokens(stop_token_ids, min_length=min_length)
            ])

            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_length,
                min_new_tokens=min_length,  # Add minimum tokens
                temperature=max(temperature, 1e-7),
                do_sample=temperature > 0,
                top_p=0.95,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                no_repeat_ngram_size=3,  # Prevent repetition
                repetition_penalty=1.2  # Penalize repetition
            )

            # Decode and clean responses
            responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            cleaned_responses = [self.clean_response(r) for r in responses]
            
            # Log sample of cleaned responses for debugging
            if len(cleaned_responses) > 0:
                logger.debug(f"Sample cleaned response: {cleaned_responses[0][:200]}...")

            # Re-encode cleaned responses
            cleaned_encodings = self.tokenizer(
                cleaned_responses,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            return cleaned_encodings['input_ids']

        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise RuntimeError("Text generation failed.") from e


class RewardsDict(TypedDict):
    """
    TypedDict for rewards and related metrics.
    """
    rewards: torch.Tensor
    bleu: List[float]
    rouge: List[Dict[str, float]]
    cyclomatic: List[float]


class SCoReTrainer:
    """
    Trainer class for the SCoRe system.
    """

    def __init__(
        self,
        model: AdvancedModel,
        ref_model: AdvancedModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config
    ):
        self.task = config.task
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.global_step = 0
        self.reward_history: List[float] = []
        self.edit_distance_ratios: List[float] = []
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision and torch.cuda.is_available())
        if config.task == 'MATH':
            self.rouge = Rouge()
            self.smoothing = SmoothingFunction()

    def compute_kl_divergence(self, logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between model logits and reference logits.

        Args:
            logits (torch.Tensor): Logits from the model.
            ref_logits (torch.Tensor): Logits from the reference model.

        Returns:
            torch.Tensor: KL divergence loss.
        """
        try:
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            ref_probs = nn.functional.softmax(ref_logits, dim=-1)
            kl_div = self.kl_loss_fn(log_probs, ref_probs)
            return kl_div
        except Exception as e:
            logger.error(f"Error computing KL divergence: {e}")
            raise RuntimeError("KL divergence computation failed.") from e

    def reward_function_math(self, generated: str, correct: str) -> Tuple[float, float, Dict[str, float]]:
        """
        Compute rewards for math tasks.

        Args:
            generated (str): Generated answer.
            correct (str): Correct answer.

        Returns:
            Tuple containing reward, BLEU score, and ROUGE scores.
        """
        # Initialize default values
        reward = 0.0
        bleu = 0.0
        rouge = {}

        logger.info(f"\n=== Math Reward Computation ===")
        trace_info = {
            "generated_answer": {
                "raw": generated,
                "boxed": None,
                "cleaned": None,
                "parsed_expr": None
            },
            "correct_answer": {
                "raw": correct,
                "boxed": None,
                "cleaned": None,
                "parsed_expr": None
            },
            "reward_computation": {
                "method_used": None,
                "final_reward": 0.0,
                "comparison_result": None,
                "error": None
            },
            "metrics": {
                    "bleu": 0.0,
                    "rouge": {}
                }
        }
        try:
            # Extract answer from \boxed{} format
            def extract_boxed_answer(text: str) -> Optional[str]:
                import re
                # More flexible pattern to handle multi-line and nested braces
                pattern = r'\\boxed{([^{}]*(?:{[^{}]*})*[^{}]*)}'
                match = re.search(pattern, text)
                if match:
                    return match.group(1).strip()
                # If no boxed answer found, try to extract the final answer after "Final Answer:"
                final_answer_pattern = r'Final Answer:.*?\\boxed{([^{}]*(?:{[^{}]*})*[^{}]*)}'
                final_match = re.search(final_answer_pattern, text, re.DOTALL)
                if final_match:
                    return final_match.group(1).strip()
                # If still no match, look for any mathematical expression
                math_pattern = r'\\left\((.*?)\\'
                math_match = re.search(math_pattern, text)
                if math_match:
                    return math_match.group(1).strip()
                return None

            # Clean and extract answers
            generated_ans = extract_boxed_answer(generated)
            correct_ans = extract_boxed_answer(correct)

            logger.info(f"After boxed extraction:")
            logger.info(f"Generated: {generated_ans}")
            logger.info(f"Correct: {correct_ans}")

            if generated_ans is None:
                logger.info("No boxed answer found in generated answer")
                return 0.0, 0.0, {'f': 0.0}

            # Clean the answers
            for char in [' ', '$', '\\n', '\\', '{', '}']:
                generated_ans = generated_ans.replace(char, '')
                correct_ans = correct_ans.replace(char, '')
            
            generated_ans = generated_ans.lower().strip()
            correct_ans = correct_ans.lower().strip()

            logger.info(f"After cleaning:")
            logger.info(f"Generated: {generated_ans}")
            logger.info(f"Correct: {correct_ans}")

            trace_info["generated_answer"]["cleaned"] = generated_ans
            trace_info["correct_answer"]["cleaned"] = correct_ans

            # Try parsing expressions
            try:
                logger.info("Attempting to parse mathematical expressions...")
                gen_expr = parse_expr(generated_ans)
                cor_expr = parse_expr(correct_ans)
                logger.info(f"Parsed generated: {gen_expr}")
                logger.info(f"Parsed correct: {cor_expr}")
                
                difference = simplify(gen_expr - cor_expr)
                logger.info(f"Simplified difference: {difference}")
                
                trace_info["generated_answer"]["parsed_expr"] = str(gen_expr)
                trace_info["correct_answer"]["parsed_expr"] = str(cor_expr)
                trace_info["reward_computation"]["method_used"] = "symbolic_comparison"
                trace_info["reward_computation"]["comparison_result"] = str(difference)
                
                reward = 1.0 if difference == 0 else 0.0
                logger.info(f"Expression comparison reward: {reward}")
            except (SympifyError, TypeError, ValueError) as e:
                logger.info(f"Expression parsing failed: {str(e)}")
                logger.info("Falling back to string comparison")
                
                trace_info["reward_computation"]["method_used"] = "string_comparison"
                trace_info["reward_computation"]["error"] = str(e)
                reward = 1.0 if generated_ans == correct_ans else 0.0
                logger.info(f"String comparison reward: {reward}")

            # Compute BLEU score if enabled
            if self.config.compute_bleu:
                try:
                    bleu = sentence_bleu(
                        [correct.split()],
                        generated.split(),
                        smoothing_function=self.smoothing.method1
                    )
                    trace_info["metrics"]["bleu"] = bleu
                except Exception as e:
                    logger.error(f"Error computing BLEU score: {e}")
                    bleu = 0.0
                    trace_info["metrics"]["bleu"] = bleu

            # Compute ROUGE score if enabled
            if self.config.compute_rouge:
                try:
                    rouge_scores = self.rouge.get_scores(generated, correct)[0]
                    rouge = {'f': rouge_scores.get('f', 0.0)}
                    trace_info["metrics"]["rouge"] = rouge_scores
                except Exception as e:
                    logger.error(f"Error computing ROUGE score: {e}")
                    trace_info["metrics"]["rouge"] = {"error": str(e)}
                    rouge = {'f': 0.0}

        except Exception as e:
            trace_info["reward_computation"]["error"] = str(e)
            logger.error(f"Error in reward computation: {e}")
            reward = 0.0
            bleu = 0.0
            rouge = {'f': 0.0}

        logger.info(f"=== Final Reward Metrics ===")
        logger.info(f"Reward: {reward}")
        logger.info(f"BLEU: {bleu}")
        logger.info(f"ROUGE: {rouge}")
        logger.info("===========================\n")   
        self._save_trace(trace_info)
        return reward, bleu, rouge

    def _save_trace(self, trace_info: Dict) -> None:
        """
        Save trace information to a JSON file with pretty printing.
        """
        try:
            trace_file = os.path.join(self.config.output_dir, 'reward_traces.jsonl')
            with open(trace_file, 'a') as f:
                # Pretty print the JSON with indentation
                json_str = json.dumps(trace_info, indent=2)
                # Add a newline after each JSON object
                f.write(json_str + '\n\n')
        except Exception as e:
            logger.error(f"Error saving trace information: {e}")

    def safe_execute_code(self, code: str, test: str, timeout: int = 5) -> bool:
        """
        Safely execute generated code with a test case.

        Args:
            code (str): Generated code.
            test (str): Test case code.
            timeout (int): Timeout in seconds.

        Returns:
            bool: Execution success status.
        """
        def target(exec_globals: Dict[str, Any]) -> None:
            try:
                exec(code, exec_globals)
                exec(test, exec_globals)
                exec_globals['exec_success'] = True
            except Exception as e:
                logger.warning(f"Execution error: {e}")
                exec_globals['exec_success'] = False

        exec_globals: Dict[str, Any] = {}
        thread = threading.Thread(target=target, args=(exec_globals,), daemon=True)
        try:
            thread.start()
            thread.join(timeout)
            success = exec_globals.get('exec_success', False)
            if not success and thread.is_alive():
                logger.warning("Code execution timed out.")
                return False
            return success
        except Exception as e:
            logger.error(f"Error during code execution thread: {e}")
            return False

    def compute_cyclomatic_complexity(self, code: str) -> float:
        """
        Compute cyclomatic complexity of the given code.

        Args:
            code (str): Code to analyze.

        Returns:
            float: Average cyclomatic complexity.
        """
        try:
            complexity = radon_complexity.cc_visit(code)
            avg_complexity = np.mean([block.complexity for block in complexity]) if complexity else 0.0
            logger.debug(f"Cyclomatic complexity: {avg_complexity}")
            return avg_complexity
        except SyntaxError as e:
            logger.warning(f"SyntaxError while computing cyclomatic complexity: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error computing cyclomatic complexity: {e}")
            return 0.0

    def reward_function_code(self, code: str, test: str) -> Tuple[float, float]:
        """
        Compute rewards for code tasks.

        Args:
            code (str): Generated code.
            test (str): Test case code.

        Returns:
            Tuple containing reward and cyclomatic complexity.
        """
        success = self.safe_execute_code(code, test)
        cyclomatic = self.compute_cyclomatic_complexity(code) if self.config.compute_cyclomatic_complexity else 0.0
        reward = 1.0 if success else 0.0
        logger.debug(f"Code reward: {reward}, Cyclomatic complexity: {cyclomatic}")
        return reward, cyclomatic

    def compute_rewards(
        self,
        generated: List[str],
        correct: List[str],
        test_cases: Optional[List[str]]
    ) -> RewardsDict:
        """
        Compute rewards for a batch of generated outputs.

        Args:
            generated (List[str]): List of generated outputs.
            correct (List[str]): List of correct answers or code.
            test_cases (Optional[List[str]]): List of test cases for code tasks.

        Returns:
            RewardsDict: Dictionary containing rewards and metrics.
        """
        rewards = []
        bleu = []
        rouge = []
        cyclomatic = []

        for i, gen in enumerate(generated):
            try:
                if self.config.task == 'MATH':
                    r, b, ro = self.reward_function_math(gen, correct[i])
                    rewards.append(r)
                    bleu.append(b)
                    rouge.append(ro)
                elif self.config.task == 'CODE':
                    test = test_cases[i] if test_cases and i < len(test_cases) else ''
                    if test:
                        r, c = self.reward_function_code(gen, test)
                    else:
                        logger.warning(f"Missing test case for CODE task at index {i}. Assigning zero reward.")
                        r, c = 0.0, 0.0
                    rewards.append(r)
                    cyclomatic.append(c)
            except Exception as e:
                logger.error(f"Error computing rewards for index {i}: {e}")
                rewards.append(0.0)
                if self.config.task == 'MATH':
                    bleu.append(0.0)
                    rouge.append({})
                elif self.config.task == 'CODE':
                    cyclomatic.append(0.0)

        rewards_tensor = torch.tensor(rewards, device=self.config.device)
        logger.debug(f"Rewards computed: {rewards}")
        return {
            'rewards': rewards_tensor,
            'bleu': bleu,
            'rouge': rouge,
            'cyclomatic': cyclomatic
        }

    def compute_edit_distance_ratio(self, s1: str, s2: str) -> float:
        """
        Compute the edit distance ratio between two strings.

        Args:
            s1 (str): First string.
            s2 (str): Second string.

        Returns:
            float: Edit distance ratio.
        """
        try:
            ratio = SequenceMatcher(None, s1, s2).ratio()
            logger.debug(f"Edit distance ratio between '{s1}' and '{s2}': {ratio}")
            return ratio
        except Exception as e:
            logger.error(f"Error computing edit distance ratio: {e}")
            return 0.0

    def prepare_batch(
        self,
        batch: List[Dict[str, Any]],
        turn: int = 1,
        prev_attempts: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str], Optional[List[str]]]:
        """
        Prepare a batch of data for processing.
        """
        try:
            if self.task == 'MATH':
                # Fix: Properly access the problem from batch dictionary
                if isinstance(batch, dict):
                    problem = batch['problem']
                    correct = batch['solution']
                else:
                    # Handle case where batch is a list
                    problem = [item['problem'] for item in batch]
                    correct = [item['solution'] for item in batch]
                    
                if turn == 1:
                    inputs = get_math_first_turn_prompt(problem) if isinstance(problem, str) else [get_math_first_turn_prompt(p) for p in problem]
                else:
                    inputs = get_math_correction_prompt(problem, prev_attempts) if isinstance(problem, str) else [get_math_correction_prompt(p, pa) for p, pa in zip(problem, prev_attempts)]
                tests = None
            elif self.task == 'CODE':
                if isinstance(batch, dict):
                    problem = batch.get('text', batch.get('prompt', ''))
                    correct = batch.get('code', batch.get('canonical_solution', ''))
                    tests = batch.get('test_list', batch.get('test', ''))
                else:
                    problem = [item.get('text', item.get('prompt', '')) for item in batch]
                    correct = [item.get('code', item.get('canonical_solution', '')) for item in batch]
                    tests = [item.get('test_list', item.get('test', '')) for item in batch]
                
                if turn == 1:
                    inputs = get_code_first_turn_prompt(problem) if isinstance(problem, str) else [get_code_first_turn_prompt(p) for p in problem]
                else:
                    inputs = get_code_correction_prompt(problem, prev_attempts) if isinstance(problem, str) else [get_code_correction_prompt(p, pa) for p, pa in zip(problem, prev_attempts)]
            else:
                # Handle other tasks
                raise NotImplementedError("Not implemented for this task")
                
            logger.debug(f"Batch prepared with {len(inputs)} samples.")
            return inputs, correct, tests
        except Exception as e:
            logger.error(f"Error preparing batch: {e}")
            raise RuntimeError("Failed to prepare batch.") from e

    def train(self) -> None:
        """
        Train the model through both training stages.
        """
        try:
            logger.info("Starting training process.")
            for epoch in range(self.config.num_epochs_stage_one):
                logger.info(f"Starting Stage I Training - Epoch {epoch + 1}")
                self.stage_one()
            for epoch in range(self.config.num_epochs_stage_two):
                logger.info(f"Starting Stage II Training - Epoch {epoch + 1}")
                self.stage_two()
            logger.info("Training completed successfully.")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def stage_one(self) -> None:
        """
        Stage I training: Train the model to produce high-reward responses at the second attempt
        while constraining first attempt to be close to base model.
        """
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Stage I Training"):
            self.global_step += 1
            try:
                inputs, correct, tests = self.prepare_batch(batch, turn=1)
                
                # First attempt: Constrain to base model
                first_encodings = self.model.tokenizer(
                    inputs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_len
                ).to(self.config.device)
                
                # Get logits for first attempt
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision and torch.cuda.is_available()):
                    first_logits = self.model(first_encodings['input_ids'], first_encodings['attention_mask'])
                    with torch.no_grad():
                        ref_logits = self.ref_model(first_encodings['input_ids'], first_encodings['attention_mask'])
                    
                    # KL divergence on first attempt to stay close to base model
                    kl_loss = self.compute_kl_divergence(first_logits, ref_logits) * self.config.beta_2
                    
                    # Generate first attempt response
                    first_ids = self.model.generate_text(first_encodings, max_length=self.config.max_seq_len, temperature=0.7)
                    first_responses = self.model.tokenizer.batch_decode(first_ids, skip_special_tokens=True)
                    
                    # Print first attempt details
                    if self.global_step % self.config.logging_steps == 0:
                        for idx, (inp, resp, corr) in enumerate(zip(inputs, first_responses, correct)):
                            logger.info(f"\n=== Sample {idx + 1} First Attempt ===")
                            logger.info(f"Input:\n{inp}")
                            logger.info(f"Model Response:\n{resp}")
                            logger.info(f"Correct Answer:\n{corr}")
                    
                    # Create second attempt inputs
                    second_inputs, correct, tests = self.prepare_batch(
                        batch, 
                        turn=2,
                        prev_attempts=first_responses
                    )
                    
                    # Second attempt: Optimize for high reward
                    second_encodings = self.model.tokenizer(
                        second_inputs,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_seq_len
                    ).to(self.config.device)
                    
                    second_ids = self.model.generate_text(second_encodings, max_length=self.config.max_seq_len)
                    second_responses = self.model.tokenizer.batch_decode(second_ids, skip_special_tokens=True)
                    
                    # Print second attempt details
                    if self.global_step % self.config.logging_steps == 0:
                        for idx, (prompt, resp) in enumerate(zip(second_inputs, second_responses)):
                            logger.info(f"\n=== Sample {idx + 1} Second Attempt ===")
                            logger.info(f"Prompt:\n{prompt}")
                            logger.info(f"Model Response:\n{resp}")
                    
                    # Compute rewards
                    rewards = self.compute_rewards(second_responses, correct, tests)['rewards']
                    
                    # Print rewards
                    if self.global_step % self.config.logging_steps == 0:
                        logger.info(f"\nRewards: {rewards.tolist()}")
                    
                    # Total loss is negative reward for second attempt plus KL penalty on first attempt
                    loss = -rewards.mean() + kl_loss

            except Exception as e:
                logger.error(f"Error during Stage I forward pass: {e}")
                continue

            try:
                # Optimization step
                self.optimizer.zero_grad()
                if self.config.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()

            except Exception as e:
                logger.error(f"Error during Stage I backward pass: {e}")
                continue

            if self.global_step % self.config.logging_steps == 0:
                logger.info(f"Stage I - Step {self.global_step}, Loss: {loss.item():.4f}")

    def stage_two(self) -> None:
        """
        Stage II training: Jointly optimize both attempts with reward shaping
        to prevent collapse to non-correcting behavior.
        """
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Stage II Training"):
            self.global_step += 1
            try:
                inputs, correct, tests = self.prepare_batch(batch, turn=1)
                
                # First attempt
                first_encodings = self.model.tokenizer(
                    inputs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_len
                ).to(self.config.device)
                
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision and torch.cuda.is_available()):
                    # Generate first attempt
                    first_ids = self.model.generate_text(first_encodings, max_length=self.config.max_seq_len, temperature=0.7)
                    first_responses = self.model.tokenizer.batch_decode(first_ids, skip_special_tokens=True)
                    first_rewards = self.compute_rewards(first_responses, correct, tests)['rewards']
                    
                    # Print first attempt details
                    if self.global_step % self.config.logging_steps == 0:
                        for idx, (inp, resp, corr) in enumerate(zip(inputs, first_responses, correct)):
                            logger.info(f"\n=== Sample {idx + 1} First Attempt ===")
                            logger.info(f"Input:\n{inp}")
                            logger.info(f"Model Response:\n{resp}")
                            logger.info(f"Correct Answer:\n{corr}")
                    
                    # Second attempt with self-correction instruction
                    second_inputs, correct, tests = self.prepare_batch(
                        batch, 
                        turn=2,
                        prev_attempts=first_responses
                    )
                    
                    second_encodings = self.model.tokenizer(
                        second_inputs,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_seq_len
                    ).to(self.config.device)
                    
                    second_ids = self.model.generate_text(second_encodings, max_length=self.config.max_seq_len)
                    second_responses = self.model.tokenizer.batch_decode(second_ids, skip_special_tokens=True)
                    second_rewards = self.compute_rewards(second_responses, correct, tests)['rewards']
                    
                    if self.global_step % self.config.logging_steps == 0:
                        for idx, (prompt, resp) in enumerate(zip(second_inputs, second_responses)):
                            logger.info(f"\n=== Sample {idx + 1} Second Attempt ===")
                            logger.info(f"Prompt:\n{prompt}")
                            logger.info(f"Model Response:\n{resp}")

                    # Compute reward bonus for making progress
                    progress_bonus = self.config.alpha * (second_rewards - first_rewards)
                    
                    # Total reward is sum of both attempts plus progress bonus
                    total_rewards = first_rewards + second_rewards + progress_bonus
                    
                    # KL regularization for both attempts
                    first_logits = self.model(first_encodings['input_ids'], first_encodings['attention_mask'])
                    second_logits = self.model(second_encodings['input_ids'], second_encodings['attention_mask'])
                    with torch.no_grad():
                        first_ref_logits = self.ref_model(first_encodings['input_ids'], first_encodings['attention_mask'])
                        second_ref_logits = self.ref_model(second_encodings['input_ids'], second_encodings['attention_mask'])
                    
                    kl_loss = (self.compute_kl_divergence(first_logits, first_ref_logits) + 
                            self.compute_kl_divergence(second_logits, second_ref_logits)) * self.config.beta_1
                    
                    # Final loss
                    loss = -total_rewards.mean() + kl_loss

            except Exception as e:
                logger.error(f"Error during Stage II forward pass: {e}")
                continue

            try:
                # Optimization step
                self.optimizer.zero_grad()
                if self.config.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()

            except Exception as e:
                logger.error(f"Error during Stage II backward pass: {e}")
                continue

            if self.global_step % self.config.logging_steps == 0:
                logger.info(
                    f"Stage II - Step {self.global_step}, Loss: {loss.item():.4f}, "
                    f"Total Reward: {total_rewards.mean().item():.4f}"
                )
    def evaluate(self) -> None:
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        total_correct_t1, total_correct_t2, total_samples = 0.0, 0.0, 0
        delta_i_to_c, delta_c_to_i = 0, 0
        bleu_scores, rouge_scores, cyclomatic_complexities = [], [], []

        try:
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Evaluation"):
                    try:
                        inputs, correct, tests = self.prepare_batch(batch, turn=1)
                        encodings = self.model.tokenizer(
                            inputs,
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=self.config.max_seq_len
                        ).to(self.config.device)
                    except Exception as e:
                        logger.error(f"Error during batch encoding in evaluation: {e}")
                        continue

                    try:
                        # Generate first attempt
                        first_ids = self.model.generate_text(encodings, max_length=self.config.max_seq_len, temperature=0.7)
                        first = self.model.tokenizer.batch_decode(first_ids, skip_special_tokens=True)
                        # Generate second attempt based on first
                        second_inputs, correct, tests = self.prepare_batch(
                            batch,
                            turn=2,
                            prev_attempts=first
                        )
                        second_encodings = self.model.tokenizer(
                            second_inputs,
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=self.config.max_seq_len
                        ).to(self.config.device)
                        second_ids = self.model.generate_text(second_encodings, max_length=self.config.max_seq_len, temperature=0.7)
                        second = self.model.tokenizer.batch_decode(second_ids, skip_special_tokens=True)
                        # Compute rewards
                        if self.global_step % self.config.logging_steps == 0:
                            for idx, (inp, f_resp, s_resp, corr) in enumerate(zip(inputs, first, second, correct)):
                                logger.info(f"\n=== Sample {idx + 1} ===")
                                logger.info(f"Problem:\n{batch[idx]['problem']}")
                                logger.info(f"First attempt:\n{f_resp}")
                                logger.info(f"Second attempt:\n{s_resp}")
                                logger.info(f"Correct answer:\n{corr}")
                        rewards_first = self.compute_rewards(first, correct, tests)['rewards']
                        rewards_second = self.compute_rewards(second, correct, tests)['rewards']
                    except Exception as e:
                        logger.error(f"Error during text generation or reward computation in evaluation: {e}")
                        continue

                    for i in range(len(inputs)):
                        try:
                            r1 = rewards_first[i].item()
                            r2 = rewards_second[i].item()
                            total_correct_t1 += r1
                            total_correct_t2 += r2
                            if r1 == 0 and r2 == 1:
                                delta_i_to_c += 1
                            elif r1 == 1 and r2 == 0:
                                delta_c_to_i += 1
                            total_samples += 1

                            if self.config.task == 'MATH':
                                if self.config.compute_bleu:
                                    bleu_first = self.compute_rewards([first[i]], [correct[i]], tests)['bleu'][0]
                                    bleu_second = self.compute_rewards([second[i]], [correct[i]], tests)['bleu'][0]
                                    bleu_scores.extend([bleu_first, bleu_second])
                                if self.config.compute_rouge:
                                    rouge_first = self.compute_rewards([first[i]], [correct[i]], tests)['rouge'][0].get('f', 0.0)
                                    rouge_second = self.compute_rewards([second[i]], [correct[i]], tests)['rouge'][0].get('f', 0.0)
                                    rouge_scores.extend([rouge_first, rouge_second])
                            elif self.config.task == 'CODE':
                                if self.config.compute_cyclomatic_complexity:
                                    cyclomatic = self.compute_rewards([second[i]], [correct[i]], tests)['cyclomatic'][0]
                                    cyclomatic_complexities.append(cyclomatic)

                            # Compute edit distance ratio
                            ratio = self.compute_edit_distance_ratio(first[i], second[i])
                            self.edit_distance_ratios.append(ratio)
                        except Exception as e:
                            logger.error(f"Error during evaluation metrics computation for sample {i}: {e}")

            # Compute final metrics
            accuracy_t1 = total_correct_t1 / total_samples if total_samples > 0 else 0.0
            accuracy_t2 = total_correct_t2 / total_samples if total_samples > 0 else 0.0
            delta = accuracy_t2 - accuracy_t1
            delta_i_to_c_frac = delta_i_to_c / total_samples if total_samples > 0 else 0.0
            delta_c_to_i_frac = delta_c_to_i / total_samples if total_samples > 0 else 0.0

            logger.info(f"Accuracy@t1: {accuracy_t1:.4f}")
            logger.info(f"Accuracy@t2: {accuracy_t2:.4f}")
            logger.info(f"Δ(t1,t2): {delta:.4f}")
            logger.info(f"Δ_i→c(t1,t2): {delta_i_to_c_frac:.4f}")
            logger.info(f"Δ_c→i(t1,t2): {delta_c_to_i_frac:.4f}")

            if self.config.task == 'MATH':
                if self.config.compute_bleu and bleu_scores:
                    avg_bleu = np.mean(bleu_scores)
                    logger.info(f"Average BLEU Score: {avg_bleu:.4f}")
                if self.config.compute_rouge and rouge_scores:
                    avg_rouge = np.mean([score for score in rouge_scores if score is not None])
                    logger.info(f"Average ROUGE-F1 Score: {avg_rouge:.4f}")
            elif self.config.task == 'CODE':
                if self.config.compute_cyclomatic_complexity and cyclomatic_complexities:
                    avg_cyclomatic = np.mean(cyclomatic_complexities)
                    logger.info(f"Average Cyclomatic Complexity: {avg_cyclomatic:.4f}")

            self.plot_reward_history()
            self.plot_edit_distance_ratios()

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

    def plot_reward_history(self) -> None:
        """
        Plot and save the training reward history.
        """
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.reward_history, label='Average Reward')
            plt.xlabel('Training Steps')
            plt.ylabel('Average Reward')
            plt.title('Training Reward Over Time')
            plt.legend()
            plt.tight_layout()
            reward_path = os.path.join(self.config.output_dir, 'training_reward.png')
            plt.savefig(reward_path)
            plt.close()
            logger.info(f"Saved reward history plot to {reward_path}.")
        except Exception as e:
            logger.error(f"Error plotting reward history: {e}")

    def plot_edit_distance_ratios(self) -> None:
        """
        Plot and save the histogram of edit distance ratios.
        """
        try:
            plt.figure(figsize=(10, 5))
            plt.hist(self.edit_distance_ratios, bins=50, color='skyblue', edgecolor='black')
            plt.xlabel('Edit Distance Ratio')
            plt.ylabel('Frequency')
            plt.title('Edit Distance Ratios between Attempts')
            plt.tight_layout()
            edit_distance_path = os.path.join(self.config.output_dir, 'edit_distance_ratios.png')
            plt.savefig(edit_distance_path)
            plt.close()
            logger.info(f"Saved edit distance ratios plot to {edit_distance_path}.")
        except Exception as e:
            logger.error(f"Error plotting edit distance ratios: {e}")


def main():
    """
    Main function to parse arguments and initiate training and evaluation.
    """
    parser = argparse.ArgumentParser(description="Advanced SCoRe System with Enhanced Features")
    parser.add_argument('--task', type=str, default='MATH', choices=['MATH', 'CODE'], help="Task type: MATH or CODE")
    parser.add_argument('--model_variant', type=str, default='meta-llama/Llama-3.2-1B-Instruct', help="Model variant to use")
    parser.add_argument('--ablation', type=str, default='none', help="Ablation setting")
    parser.add_argument('--data_path', type=str, default='./data', help="Path to the data directory")
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Directory to save outputs")
    parser.add_argument('--mixed_precision', action='store_true', help="Enable mixed precision training")
    parser.add_argument('--no_bleu', action='store_false', dest='compute_bleu', help="Disable BLEU score computation")
    parser.add_argument('--no_rouge', action='store_false', dest='compute_rouge', help="Disable ROUGE score computation")
    parser.add_argument('--no_cyclomatic', action='store_false', dest='compute_cyclomatic_complexity', help="Disable cyclomatic complexity computation")
    args = parser.parse_args()

    # Initialize configuration
    config = Config(
        task=args.task,
        model_variant=args.model_variant,
        ablation=args.ablation,
        data_path=args.data_path,
        output_dir=args.output_dir,
        mixed_precision=args.mixed_precision,
        compute_bleu=args.compute_bleu,
        compute_rouge=args.compute_rouge,
        compute_cyclomatic_complexity=args.compute_cyclomatic_complexity,
        logging_steps=1
    )

    try:
        config.validate()
    except Exception as e:
        logger.critical(f"Configuration validation failed: {e}")
        return

    try:
        set_seed(config.seed)
    except Exception as e:
        logger.critical(f"Failed to set seed: {e}")
        return

    # Determine data files based on task
    if config.task == 'MATH':
        train_file = os.path.join(config.data_path, 'selected_problems_105.json')
        val_file = os.path.join(config.data_path, 'math_test.json')
    elif config.task == 'CODE':
        train_file = os.path.join(config.data_path, 'mbpp_train.json')
        val_file = os.path.join(config.data_path, 'HumanEval.jsonl')
    else:
        logger.critical("Invalid task specified. Choose between 'MATH' and 'CODE'.")
        return

    # Check data file existence
    for file in [train_file, val_file]:
        if not os.path.isfile(file):
            logger.critical(f"Data file {file} does not exist.")
            return

    # Load datasets
    try:
        if config.task == 'MATH':
            train_data = load_json(train_file, 1000)
            val_data = load_json(val_file, 100)
        elif config.task == 'CODE':
            train_data = load_json(train_file, 1000)
            val_data = load_json(val_file, 100)
        train_dataset = BaseDataset(train_data)
        val_dataset = BaseDataset(val_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        logger.info("Datasets loaded successfully.")
    except Exception as e:
        logger.critical(f"Error loading data: {e}")
        return

    # Initialize models
    try:
        model = AdvancedModel(config.model_variant, config.device)
        ref_model = AdvancedModel(config.model_variant, config.device)
        ref_model.model.eval()
        for param in ref_model.model.parameters():
            param.requires_grad = False
        logger.info("Models initialized successfully.")
    except Exception as e:
        logger.critical(f"Error initializing models: {e}")
        return

    # Setup optimizer and scheduler
    try:
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
        total_steps = len(train_loader) * (config.num_epochs_stage_one + config.num_epochs_stage_two)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        logger.info("Optimizer and scheduler set up successfully.")
    except Exception as e:
        logger.critical(f"Error setting up optimizer and scheduler: {e}")
        return

    # Initialize trainer
    try:
        trainer = SCoReTrainer(
            model=model,
            ref_model=ref_model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        logger.info("Trainer initialized successfully.")
    except Exception as e:
        logger.critical(f"Error initializing trainer: {e}")
        return

    # Start training and evaluation
    try:
        trainer.train()
        trainer.evaluate()
    except Exception as e:
        logger.critical(f"Error during training/evaluation: {e}")
        return

    # Save the trained model
    try:
        model_save_path = os.path.join(config.output_dir, 'score_model.bin')
        torch.save(model.model.state_dict(), model_save_path)
        logger.info(f"Model saved to {model_save_path}.")
    except Exception as e:
        logger.critical(f"Error saving the model: {e}")
        return


def load_model(model_path: str, model_variant: str, device: torch.device) -> AdvancedModel:
    """
    Load a saved model from disk.

    Args:
        model_path (str): Path to the saved model state dict.
        model_variant (str): Model variant identifier.
        device (torch.device): Device to load the model onto.

    Returns:
        AdvancedModel: Loaded model instance.
    """
    try:
        advanced_model = AdvancedModel(model_variant, device)
        advanced_model.model.load_state_dict(torch.load(model_path, map_location=device))
        advanced_model.model.to(device)
        advanced_model.model.eval()
        logger.info(f"Model loaded from {model_path} and moved to {device}.")
        return advanced_model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}") from e
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise RuntimeError(f"Failed to load model from {model_path}") from e


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        raise