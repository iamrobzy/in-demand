import requests

def show_examples(n = 10):

    url = f"https://datasets-server.huggingface.co/rows?dataset=jjzha%2Fskillspan&config=default&split=train&offset=0&length={n}"
    response = requests.get(url)

    if response.status_code == 200:

        data = response.json()
        for i in range(n):
            row = data['rows'][i]['row']
            tokens = row['tokens']
            skill_labels, knowledge_labels = row['tags_skill'], row['tags_knowledge']

            print(f'Example #{i+1}')
            print('Tokens:', tokens)
            print('Skill Labels:', skill_labels)
            print('Knowledge Labels:', knowledge_labels)
            print('')


show_examples(n=100)