import requests
import os
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def show_examples(n = 10):

    url = f"https://datasets-server.huggingface.co/rows?dataset=jjzha%2Fskillspan&config=default&split=train&offset=0&length={n}"
    response = requests.get(url)

    if response.status_code == 200:

        data = response.json()
        for i in range(n):
            row = data['rows'][i]['row']
            tokens = row['tokens']
            skill_labels, knowledge_labels = row['tags_skill'], row['tags_knowledge']

            with open(f"{repo_dir}/examples.txt", 'w') as file:
                file.write(f'Example #{i+1}\n')
                file.write(f'Tokens: {str(tokens)}\n')
                file.write(f'Skill Labels: {str(skill_labels)}\n')
                file.write(f'Knowledge Labels: {str(knowledge_labels)}\n')
                file.write('\n')


show_examples(n=100)