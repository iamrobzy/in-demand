import requests
import os
repo_dir = os.getcwd()
print(repo_dir)

def show_examples(n = 10):

    url = f"https://datasets-server.huggingface.co/rows?dataset=jjzha%2Fskillspan&config=default&split=train&offset=0&length={n}"
    response = requests.get(url)

    if response.status_code == 200:

        data = response.json()

        tags_knowledge = [str(a['row']['tags_knowledge']) for a in data['rows']]
        tokens = [str(a['row']['tokens']) for a in data['rows']]

        with open(f"{repo_dir}/few_shot.txt", 'w') as file:
            for i in range(n):
                file.write(f'tags_knowledge: {tags_knowledge[i]}\n')
                file.write(f'tokens: {tokens[i]}\n')
                file.write('\n')


show_examples(n=100)