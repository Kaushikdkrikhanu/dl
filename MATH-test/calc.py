import json

def calculate_accuracy(data):
    total_questions = len(data['answer_pairs'])
    first_correct = sum(1 for pair in data['answer_pairs'] if pair['first_answer_correct'])
    second_correct = sum(1 for pair in data['answer_pairs'] if pair['second_answer_correct'])
    
    first_accuracy = (first_correct / total_questions) * 100
    second_accuracy = (second_correct / total_questions) * 100
    
    print(f"Total questions: {total_questions}")
    print(f"First answer accuracy: {first_correct}/{total_questions} ({first_accuracy:.1f}%)")
    print(f"Second answer accuracy: {second_correct}/{total_questions} ({second_accuracy:.1f}%)")
    
    # List questions where both answers were incorrect
    print("\nQuestions where both answers were incorrect:")
    for i, pair in enumerate(data['answer_pairs'], 1):
        if not pair['first_answer_correct'] and not pair['second_answer_correct']:
            print(f"Question {i}")

# Read and parse the JSON file
with open('answer_pairs.json', 'r') as f:
    data = json.load(f)

calculate_accuracy(data)