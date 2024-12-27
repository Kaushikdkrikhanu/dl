# SCoRe (Self-Correction via Reinforcement Learning)
This is our attempt to implement SCoRe according to google's SCoRe paper.

# To ru.n score:

## Installation
pip install -r requirements.txt

## Huggingface setup
'''
huggingface-cli login

Token :  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
'''


## Run
Use this command


'''
python run.py --task MATH --model_variant meta-llama/Llama-3.2-3B-Instruct --data_path ./data --output_dir ./outputs --mixed_precision --no_bleu --no_rouge
'''

## References

https://github.com/sanowl/Self-Correcting-LLM--Reinforcement-Learning-
