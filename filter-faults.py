import json

def count_mismatch(file_path):

    count_mismatch = 0 
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            data = json.loads(line)
            tokens, tags = data['tokens'], data['tags_knowledge']
            if len(tokens) != len(tags):
                count_mismatch += 1
    
    return count_mismatch

def delete_mismatched_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            data = json.loads(line)
            tokens, tags = data['tokens'], data['tags_knowledge']
            if len(tokens) == len(tags):
                file.write(line)


if __name__ == "__main__":
    file_path = 'data/tags-04-01-2025.jsonl'
    count = count_mismatch(file_path)

    if  count > 0:
        delete_mismatched_lines(file_path)
        print(f"Deleted {count} mismatched lines.")