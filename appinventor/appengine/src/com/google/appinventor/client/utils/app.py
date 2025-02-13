# Install and import MIT Deep Learning utilities
#!pip install mitdeeplearning > /dev/null 2>&1
import sys
sys.path.append('/content/drive/My Drive/ai') # Replace your_folder_name with the actual path
import lab3

import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from lion_pytorch import Lion


from openai import OpenAI
from datasets import load_dataset
from torch.utils.data import DataLoader


# Basic question-answer template
template_without_answer = "<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
template_with_answer = template_without_answer + "{answer}<end_of_turn>\n"


# Load the tokenizer for Gemma 2B
model_id = "unsloth/gemma-2-2b-it" #"google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model -- note that this may take a few minutes
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")


# How big is the tokenizer?
print(f"Vocab size: {len(tokenizer.get_vocab())}")


def create_dataloader():
    ds = load_dataset("cindyloohome/chomsky", split="train")


    n = len(ds)
    ds_test = ds.select(range(n)) # Selects all elements from 0 to n-1

    # Create a dataloader
    dataloader = DataLoader(ds_test, batch_size=1, shuffle=True)
    dataloader_test = DataLoader(ds_test, batch_size=1, shuffle=True)
    return dataloader, dataloader_test


def chat(question, max_new_tokens=32, temperature=0.7, only_answer=False):
    # 1. Construct the prompt using the template
    prompt = template_without_answer.format(question=question)

    # 2. Tokenize the text
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(input_ids)
    # 3. Feed through the model to predict the next token probabilities
    with torch.no_grad():
        outputs = model.generate(input_ids['input_ids'], do_sample=True, max_new_tokens=max_new_tokens, temperature=temperature)

    # 4. Only return the answer if only_answer is True
    output_tokens = outputs[0]
    if only_answer:
        output_tokens = output_tokens[input_ids['input_ids'].shape[1]:]

    # 5. Decode the tokens
    result = tokenizer.decode(output_tokens, skip_special_tokens=True) # TODO

    return result



prompt = template_without_answer.format(question="testing: What does MIT stand for?")
tokens = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
output = model.generate(tokens, max_new_tokens=20)
print(tokenizer.decode(output[0]))


train_loader, test_loader = create_dataloader(style="chomsky")

sample = train_loader.dataset[44]
question = sample['Instruction']
answer = sample['response']
#answer_style = sample['response_style']

print(f"Question: {question}\n\n" +
      f"Original Answer: {answer}\n\n")

# Let's try chatting with the model now to test if it works!
answer = chat(
    question ="Who is noam chomsky?",
    max_new_tokens=52,
    temperature=.8,
    only_answer=True
)

print(answer)