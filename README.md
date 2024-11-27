# To run score:

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
python run.py --task MATH --model_variant meta-llama/Llama-3.2-1B-Instruct --data_path ./data --output_dir ./outputs --mixed_precision --no_bleu --no_rouge
'''
