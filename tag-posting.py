import spacy
import re

nlp = spacy.load("en_core_web_sm")

def split_text_recursively(text):
    if '\n' not in text:
        return [text]
    parts = text.split('\n', 1)
    return [parts[0]] + split_text_recursively(parts[1])

def parse_post(path):

    # Read the file

    with open(path, 'r') as file:
        text = file.read()

    # Sentence tokenization

    str_list = split_text_recursively(text)
    str_list = [i.strip() for i in str_list]
    str_list = list(filter(None, str_list))

    count = 0
    sents = []

    for line in str_list:
        doc = nlp(line)
        for sent in doc.sents:
            print(f"{sent.text}")
            sents.append(sent.text)

    # Skill/knowledge extraction
    
    


path = './job-postings/03-01-2024/2.txt'
parse_post(path)
