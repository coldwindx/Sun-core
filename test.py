import json
import os
import sys
from loguru import logger
from tokenizers import Tokenizer
import torch
from torch.utils.data import Dataset
from transformers import BertForMaskedLM, BertTokenizer

__PATH__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(__PATH__)  
from tools import Config

config = Config()
vocab_additional = json.load(open("vocab_additional.json", 'r'))
tokenizer = BertTokenizer.from_pretrained(config["pretrain"]["bert_pretrain_uncased"], use_fast=True)

# num_added_toks = tokenizer.add_tokens(vocab_additional, special_tokens=True)
# tokenizer = AutoTokenizer.from_pretrained("token.json")

sent_seq = [
    "redirect lookaside hh 2020 day!"
]
print(tokenizer.tokenize("redirect lookaside hh 2020 day!"))
padded_sent_seq = tokenizer(sent_seq, padding=True, truncation=True, max_length=2048, return_tensors="pt")
print(padded_sent_seq)

# print("*" * 30)
# num_added_toks = tokenizer.add_tokens(["##aside", "lookaside"], special_tokens=True)
# print(tokenizer.tokenize("redirect lookaside hh 2020 day!"))
# padded_sent_seq = tokenizer(sent_seq, padding=True, truncation=True, max_length=2048, return_tensors="pt")
# print(padded_sent_seq)