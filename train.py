from transformers import AutoTokenizer, BertForTokenClassification, TrainingArguments, Trainer
import torch
from tabulate import tabulate
import wandb


tokenizer = AutoTokenizer.from_pretrained("jjzha/jobbert_knowledge_extraction")
model = BertForTokenClassification.from_pretrained("Robzy/jobbert_knowledge_extraction")

artifact = wandb.Artifact(name="jobbert-knowledge-extraction", type="BERT")

text = 'Experience with Unreal and/or Unity and/or native IOS/Android 3D development and/or Web based 3D engines '

# Tokenize  
inputs = tokenizer(
    text, add_special_tokens=False, return_tensors="pt"
)

# Inference

# with torch.no_grad():
#     output = model(**inputs)

# # Post-process
# predicted_token_class_ids = output.logits.argmax(-1)
# predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
# tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())

# # Display
# table = zip(tokens, predicted_tokens_classes)
# print(tabulate(table, headers=["Token", "Predicted Class"], tablefmt="pretty"))

# Training

from datasets import load_dataset
dataset = load_dataset("json", data_files="data/test-short.json")


# Convert tokens to ids before training

data = [torch.tensor([tokenizer.convert_tokens_to_ids(t) for t in l]) for l in dataset['train']['tokens']]

dataset = dataset.map(
    lambda x: {"input_ids": torch.tensor(tokenizer.convert_tokens_to_ids(x["tokens"]))}
)

# Data preprocessing

from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import DataCollatorForTokenClassification
from typing import List, Tuple

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

# Training settings
batch_size = 32
train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
eval_dataloader = DataLoader(dataset['train'], batch_size=batch_size, collate_fn=collate_fn)

from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_scheduler

model.train()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

IGNORE_INDEX = -100
criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
id2label = model.config.id2label
label2id = model.config.label2id

lr = 5e-5
optimizer = AdamW(model.parameters(), lr=lr)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

model.config.pad_token_id = 0

## Training

from dotenv import load_dotenv
import os
load_dotenv(".env")

from datetime import datetime
current_time = datetime.now()

wandb.login(key=os.getenv('WANDB_API_KEY'))

run = wandb.init(
    # set the wandb project where this run will be logged
    project="in-demand",

    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "BERT",
    "epochs": num_epochs,
    "batch_size": batch_size,
    "notes": "Datetime: " + current_time.strftime("%m/%d/%Y, %H:%M:%S")
    }
)

import logging
from datetime import datetime
logging.info("Initiating training")

progress_bar = tqdm(range(num_epochs), desc="Epochs")
for epoch in range(num_epochs):
    logging.info(f"Epoch #{epoch}")
    print(f"Epoch #{epoch}")

    batch_count = 0

    for batch in train_dataloader:

        logging.info(f"Batch #{batch_count} / {len(train_dataloader)}")
        print(f"Batch #{batch_count} / {len(train_dataloader)}")

        tokens = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tags_knowledge = batch['tags_knowledge'].to(device)

        outputs = model(tokens, attention_mask=attention_mask)

        # Batch
        pred = outputs.logits.reshape(-1, model.config.num_labels) # Logits
        label = torch.where(attention_mask==0, torch.tensor(IGNORE_INDEX).to(device), tags_knowledge).reshape(-1) # Labels, padding set to class idx -100

        # Compute accuracy ignoring padding idx
        _, predicted_labels = torch.max(pred, dim=1)
        non_pad_elements = label != IGNORE_INDEX
        correct_predictions = (predicted_labels[non_pad_elements] == label[non_pad_elements]).sum().item()
        total_predictions = non_pad_elements.sum().item()
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        wandb.log({"epoch": epoch, "accuracy": accuracy, "loss": loss})

        batch_count += 1

    progress_bar.update(1)


model.push_to_hub("Robzy/jobbert_knowledge_extraction")


# Add the state_dict to the artifact
state_dict = model.state_dict()
with artifact.new_file('model.pth', mode='wb') as f:
    torch.save(state_dict, f)

# Log the artifact to W&B
wandb.log_artifact(artifact)