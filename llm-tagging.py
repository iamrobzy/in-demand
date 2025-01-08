import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import sys
from tabulate import tabulate
import spacy
import re
import json
from datetime import datetime
from tqdm import tqdm
import time


load_dotenv(".env")
nlp = spacy.load("en_core_web_sm")

def split_text_recursively(text):
    if '\n' not in text:
        return [text]
    parts = text.split('\n', 1)
    return [parts[0]] + split_text_recursively(parts[1])


def tokenize_to_sent(path):

    print(f"Tokenizing {path} to sentences...")

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
            sents.append(sent.text)

    print(f"Tokenization completed. {len(sents)} sentences found.")
    
    return sents


### LLM-based tag extraction with few-shot learning

model = ChatOpenAI(temperature=0)

class TokenTaggingResult(BaseModel):
    tokens: List[str]
    tags_knowledge: List[str]

class Results(BaseModel):
    results: List[TokenTaggingResult]


model = ChatOpenAI(model_name="gpt-4o", temperature=0.0, api_key=os.getenv('OPENAI_API_KEY'))
tokenizer = AutoTokenizer.from_pretrained("jjzha/jobbert_skill_extraction")
parser = JsonOutputParser(pydantic_object=Results)

# Definitions

skill_definition = """
Skill means the ability to apply knowledge and use know-how to complete tasks and solve problems.
"""

knowledge_definition = """
Knowledge means the outcome of the assimilation of information through learning. Knowledge is the body of facts, principles, theories and practices that is related to a field of work or study.
"""

# Few-shot examples
with open('few-shot.txt', 'r') as file:
    few_shot_examples = file.read()

prompt = PromptTemplate(
    template="""You are an expert in tagging tokens with knowledge labels. Use the following definitions to tag the input tokens:
    Knowledge definition:{knowledge_definition}
    Use the examples below to tag the input text into relevant knowledge or skills categories.\n{few_shot_examples}\n{format_instructions}\n{input}\n""",
    input_variables=["input"],
    partial_variables={"format_instructions": parser.get_format_instructions(),
                       "few_shot_examples": few_shot_examples,
                    #    "skill_definition": skill_definition,
                       "knowledge_definition": knowledge_definition},
)

def extract_tags(sents: str, tokenize = True) -> Results:

    print("Extracting tags...")
    print(f"Tokenizing {len(sents)} sentences...")

    start_time = time.time()

    if tokenize:
        tokens = [tokenizer.tokenize(t) for t in sents]

    prompt_and_model = prompt | model
    output = prompt_and_model.invoke({"input": tokens})
    output = parser.invoke(output)

    time_taken = time.time() - start_time
    print(f"Tags extracted in {time_taken} seconds.")

    return tokens, output


def tag_posting(job_path, output_path):

    # Reading & sentence tokenization
    sents = tokenize_to_sent(job_path)

    # LLM-based tag extraction
    tokens, output = extract_tags(sents, tokenize=True)

    with open(output_path, "w") as file:
        for entry in output['results']:
            json.dump(entry, file)
            file.write("\n")

def tag_all_today():

    date = datetime.today().strftime('%d-%m-%Y')
    # date = "04-01-2025"

    jobs = os.listdir(f'./job-postings/{date}')
    output_path = f'./data/tags-{date}.jsonl'
    count = 0

    for job in tqdm(jobs, desc="Tagging job postings"):

        job_path = f'./job-postings/{date}/{job}'
        
        # Reading & sentence tokenization
        sents = tokenize_to_sent(job_path)

        # LLM-based tag extraction
        tokens, output = extract_tags(sents, tokenize=True)

        with open(output_path, "a") as file:
            for entry in output['results']:
                json.dump(entry, file)
                file.write("\n")
        
        count += 1
        if count > 2:
            break
        

    print(f"Tagging completed. Output saved to {output_path}")

    
if __name__ == "__main__":

    tag_all_today()