few_shot_examples = """
Example #96
Tokens: ['Public']
Skill Labels: ['O']
Knowledge Labels: ['O']

Example #97
Tokens: ['Technologies']
Skill Labels: ['O']
Knowledge Labels: ['O']

Example #98
Tokens: ['cloud', 'java', 'amazon-web-services']
Skill Labels: ['O', 'O', 'O']
Knowledge Labels: ['B', 'B', 'B']

Example #99
Tokens: ['Job', 'description']
Skill Labels: ['O', 'O']
Knowledge Labels: ['O', 'O']

Example #100
Tokens: ['As', 'a', 'member', 'of', 'our', 'Software', 'Engineering', 'Group', 'we', 'look', 'first', 'and', 'foremost', 'for', 'people', 'who', 'are', 'passionate', 'about', 'solving', 'business', 'problems', 'through', 'innovation', 'and', 'engineering', 'practices', '.']
Skill Labels: ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'O']
Knowledge Labels: ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
"""


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

load_dotenv(".env")
# ChatOpenAI.api_key = OPENAI_API_KEY


### LLM-based tag extraction with few-shot learning

model = ChatOpenAI(temperature=0)

class TokenTaggingResult(BaseModel):
    tokens: List[str]
    skill_labels: List[str]
    knowledge_labels: List[str]


model = ChatOpenAI(model_name="gpt-4o", temperature=0.0, api_key=os.getenv('OPENAI_API_KEY'))
tokenizer = AutoTokenizer.from_pretrained("jjzha/jobbert_skill_extraction")
parser = JsonOutputParser(pydantic_object=TokenTaggingResult)

skill_definition = """
Skill means the ability to apply knowledge and use know-how to complete tasks and solve problems.
"""

knowledge_definition = """
Knowledge means the outcome of the assimilation of information through learning. Knowledge is the body of facts, principles, theories and practices that is related to a field of work or study.
"""

prompt = PromptTemplate(
    template="""You are an expert in tagging tokens with skill and knowledge labels. Use the following definitions to tag the input tokens:
    Skill definition:{skill_definition}
    Knowledge definition:{knowledge_definition}
    Use the examples below to tag the input text into relevant knowledge or skills categories.\n{few_shot_examples}\n{format_instructions}\n{input}\n""",
    input_variables=["input"],
    partial_variables={"format_instructions": parser.get_format_instructions(),
                       "few_shot_examples": few_shot_examples,
                       "skill_definition": skill_definition,
                       "knowledge_definition": knowledge_definition},
)

def extract_tags(text: str, tokenize = True) -> TokenTaggingResult:

    if tokenize:

        inputs = tokenizer(text, return_tensors="pt")
        tokens =  tokenizer.decode(inputs['input_ids'].squeeze()).split()[1:-1]

    prompt_and_model = prompt | model
    output = prompt_and_model.invoke({"input": tokens})
    output = parser.invoke(output)
    return tokens, output

### Pre-trained model from Hugging Face

mapping = {0: 'B', 1: 'I', 2: 'O'}
token_skill_classifier = AutoModelForTokenClassification.from_pretrained("jjzha/jobbert_skill_extraction")
token_knowledge_classifier = AutoModelForTokenClassification.from_pretrained("jjzha/jobbert_knowledge_extraction")

def convert(text):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        skill_outputs = token_skill_classifier(**inputs)
        knowledge_outputs = token_knowledge_classifier(**inputs)

    decoded_tokens =  tokenizer.decode(inputs['input_ids'].squeeze()).split()[1:-1]
    skill_cls = skill_outputs.logits.argmax(dim=2).squeeze()[1:-1]
    knowledge_cls = knowledge_outputs.logits.argmax(dim=2).squeeze()[1:-1]

    skill_cls = [mapping[i.item()] for i in skill_cls]
    knowledge_cls = [mapping[i.item()] for i in knowledge_cls]
    return skill_cls, knowledge_cls



if __name__ == "__main__":
    text = input('Enter text: ')

    # LLM-based tag extraction
    tokens, output = extract_tags(text, tokenize=True)

    # Pre-trained
    skill_cls, knowledge_cls = convert(text)

    table = zip(tokens, output['skill_labels'], output['knowledge_labels'], skill_cls, knowledge_cls)
    headers = ["Token", "Skill Label", "Knowledge Label", "Pred Skill Label", "Pred Knowledge Label"]
    print(tabulate(table, headers=headers, tablefmt="pretty"))