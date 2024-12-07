import json

def analyze_rewards(file_path):
    try:
        entries = []
        # Read the file line by line
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Split content by '][' to handle multiple concatenated JSON arrays
            json_parts = content.split('][')
            
            for part in json_parts:
                # Add missing brackets for middle parts
                if not part.startswith('['):
                    part = '[' + part
                if not part.endswith(']'):
                    part = part + ']'
                    
                try:
                    # Parse each part as a JSON array
                    part_entries = json.loads(part)
                    if isinstance(part_entries, list):
                        entries.extend(part_entries)
                except json.JSONDecodeError:
                    continue
        
        # Initialize counters
        total_entries = len(entries)
        first_trace_reward_1_count = 0
        first_trace_reward_0_count = 0
        second_trace_reward_1_count = 0
        second_trace_reward_0_count = 0
        
        # Process each entry
        for entry in entries:
            if isinstance(entry, dict):  # Ensure entry is a dictionary
                # Check first trace
                first_trace = entry.get('first_trace')
                if isinstance(first_trace, (int, float)):
                    # If first_trace is a number
                    reward = float(first_trace)
                elif isinstance(first_trace, dict):
                    # If first_trace is a dictionary
                    reward = first_trace.get('reward_computation', {}).get('final_reward')
                
                if reward == 1 or reward == 1.0:
                    first_trace_reward_1_count += 1
                elif reward == 0 or reward == 0.0:
                    first_trace_reward_0_count += 1
                
                # Check second trace
                second_trace = entry.get('second_trace')
                if isinstance(second_trace, (int, float)):
                    # If second_trace is a number
                    reward = float(second_trace)
                elif isinstance(second_trace, dict):
                    # If second_trace is a dictionary
                    reward = second_trace.get('reward_computation', {}).get('final_reward')
                
                if reward == 1 or reward == 1.0:
                    second_trace_reward_1_count += 1
                elif reward == 0 or reward == 0.0:
                    second_trace_reward_0_count += 1
        
        # Calculate percentages
        first_trace_total = first_trace_reward_1_count + first_trace_reward_0_count
        second_trace_total = second_trace_reward_1_count + second_trace_reward_0_count
        
        return {
            'total_entries': total_entries,
            'first_trace_reward_1_count': first_trace_reward_1_count,
            'first_trace_reward_0_count': first_trace_reward_0_count,
            'first_trace_reward_1_percent': (first_trace_reward_1_count / first_trace_total * 100) if first_trace_total > 0 else 0,
            'first_trace_reward_0_percent': (first_trace_reward_0_count / first_trace_total * 100) if first_trace_total > 0 else 0,
            'second_trace_reward_1_count': second_trace_reward_1_count,
            'second_trace_reward_0_count': second_trace_reward_0_count,
            'second_trace_reward_1_percent': (second_trace_reward_1_count / second_trace_total * 100) if second_trace_total > 0 else 0,
            'second_trace_reward_0_percent': (second_trace_reward_0_count / second_trace_total * 100) if second_trace_total > 0 else 0
        }
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def main():
    file_path = 'stage_two_logs_20241207_134942.json'
    results = analyze_rewards(file_path)
    
    if results:
        print("\nAnalysis Results:")
        print(f"Total entries analyzed: {results['total_entries']}")
        print(f"\nFirst Trace:")
        print(f"  Reward 1: {results['first_trace_reward_1_count']} ({results['first_trace_reward_1_percent']:.2f}%)")
        print(f"  Reward 0: {results['first_trace_reward_0_count']} ({results['first_trace_reward_0_percent']:.2f}%)")
        print(f"\nSecond Trace:")
        print(f"  Reward 1: {results['second_trace_reward_1_count']} ({results['second_trace_reward_1_percent']:.2f}%)")
        print(f"  Reward 0: {results['second_trace_reward_0_count']} ({results['second_trace_reward_0_percent']:.2f}%)")

if __name__ == "__main__":
    main()