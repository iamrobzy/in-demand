import spacy
import re
from transformers import AutoTokenizer, BertForTokenClassification, TrainingArguments, Trainer
import torch
from typing import List
import os


### Parsing job posting

def split_text_recursively(text):
    if '\n' not in text:
        return [text]
    parts = text.split('\n', 1)
    return [parts[0]] + split_text_recursively(parts[1])

def parse_post(path):

    nlp = spacy.load("en_core_web_sm")

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
    
    return sents


### Model inference

from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import DataCollatorForTokenClassification
from typing import List, Tuple

tokenizer = AutoTokenizer.from_pretrained("jjzha/jobbert_knowledge_extraction")
model = BertForTokenClassification.from_pretrained("Robzy/jobbert_knowledge_extraction")

id2label = model.config.id2label
label2id = model.config.label2id

def pad(list_of_lists, pad_value=0):
    max_len = max(len(lst) for lst in list_of_lists)

    # Pad shorter lists with the specified value
    padded_lists = [lst + [pad_value] * (max_len - len(lst)) for lst in list_of_lists]
    attention_masks = [[1] * len(lst) + [0] * (max_len - len(lst)) for lst in list_of_lists]
    
    return torch.tensor(padded_lists), torch.tensor(attention_masks)

def collate_fn(batch: List[List[torch.Tensor]]):

    input_ids, attention_mask = pad(list(map(lambda x: tokenizer.convert_tokens_to_ids(x['tokens']),batch)))
    tags_knowledge, _ = pad([list(map(lambda x: label2id[x],o)) for o in [b['tags_knowledge'] for b in batch]])
    return {"input_ids": input_ids, "tags_knowledge": tags_knowledge, "attention_mask": attention_mask}

def extract_spans(B_mask, I_mask, token_ids, tokenizer):
    """
    Extract text spans for 2D tensors (batch of sequences).
    """
    batch_size = B_mask.size(0)
    all_spans = []

    d = tokenizer.decode

    for batch_idx in range(batch_size):
        spans = []
        current_span = []

        for i in range(B_mask.size(1)):  # Iterate over sequence length
            if B_mask[batch_idx, i].item() == 1:  # Begin a new span
                if current_span:
                    spans.append(current_span)
                    print(d(current_span))
                current_span = [token_ids[batch_idx, i].item()]
                print(d(current_span))
            elif I_mask[batch_idx, i].item() == 1 and current_span:  # Continue the current span
                print(d(current_span))
                current_span.append(token_ids[batch_idx, i].item())
            else:  # Outside any entity
                print(d(current_span))
                if current_span:
                    spans.append(current_span)
                    current_span = []

        if current_span:  # Save the last span if it exists
            spans.append(current_span)

        # Decode spans for this sequence
        decoded_spans = [tokenizer.decode(span, skip_special_tokens=True) for span in spans]
        all_spans.append(decoded_spans)

    # Remove empty spans
    all_spans = list(filter(lambda x: x != [], all_spans))

    return all_spans


def concat_subtokens(tokens):
    result = []
    
    for token in tokens:
        if token.startswith('##'):
            # Concatenate sub-token to the last token in result
            result[-1] += token[2:]  # Remove '##' and append the continuation
        else:
            # If it's a new token, add it to result
            result.append(token)
    
    return result

def merge_spans(batch_spans, tokenizer):

    batch_decoded_spans = []

    for spans in batch_spans:

        ## Concatenate subtokens

        if spans[0].startswith('##'):
            continue

        decoded_spans = []
        for token in spans:
            if token.startswith('##'):
                # Concatenate sub-token to the last token in result
                decoded_spans[-1] += token[2:]  # Remove '##' and append the continuation
            else:
                # If it's a new token, add it to result
                decoded_spans.append(token)

        ## Concatenatation done
 
        for span in decoded_spans:
            batch_decoded_spans.append(span)

    return batch_decoded_spans


def extract_skills(batch_sentences: List[str]):

    print('Extracting skills from job posting...')

    # Batch

    # Tokenize
    batch = tokenizer(batch_sentences, padding=True, truncation=True) 
    batch_tokens = torch.tensor(batch['input_ids'])
    batch_attention_masks = torch.tensor(batch['attention_mask'])

    model.eval()
    with torch.no_grad():
        output = model(input_ids=batch_tokens, attention_mask=batch_attention_masks)

    # Post-process
    pred = output.logits.argmax(-1)
    pred = torch.where(batch_attention_masks==0, torch.tensor(-100), pred)

    b_mask = torch.where(pred==0, 1, 0)
    i_mask = torch.where(pred==1, 1, 0)

    spans = extract_spans(b_mask, i_mask, batch_tokens, tokenizer)
    decoded_spans = merge_spans(spans, tokenizer)

    return decoded_spans

def skills_save(path,skills):
    with open(path, 'w') as f:
        for i, skill in enumerate(skills):
            if i == len(skills) - 1:
                f.write(f"{skill}")
            else:
                f.write(f"{skill}\n")


def backfill():

    job_dir = os.path.join(os.getcwd(), 'job-postings')
    tag_dir = os.path.join(os.getcwd(), 'tags')

    for date in os.listdir(job_dir):
        print(f"Processing date directory: {date}")
        
        job_date = os.path.join(job_dir, date)
        tag_date = os.path.join(tag_dir, date)

        for job in os.listdir(job_date):
            job_path = os.path.join(job_date, job)
            tag_path = os.path.join(tag_date, job)

            print(f"Processing job file: {job_path}")

            if not os.path.exists(tag_date):
                os.makedirs(tag_date)
                print(f"Created directory: {tag_date}")

            sents = parse_post(job_path)
            skills = extract_skills(sents)
            skills_save(tag_path, skills)

            print(f"Saved skills to: {tag_path}")

def tag_date():

    pass

if __name__ == '__main__':

    # Backfill
    backfill()


    # path = './job-postings/03-01-2024/2.txt'
    # sents = parse_post(path)
    # skills = extract_skills(sents)
    # skills_save('./tags/03-01-2024/2.txt',skills)RAPID_API_KEY : 60a10b11e6msh821d32f6e1e955ep15b5b1jsnf61a46680409
1