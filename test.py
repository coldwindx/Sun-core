import json
import os
import sys
from loguru import logger
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

__PATH__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(__PATH__)  
from tools import Config

config = Config()
tokenizer = AutoTokenizer.from_pretrained(config["pretrain"]["bert_pretrain_uncased"])

class ScDataset(Dataset):
    def __init__(self, path):
        self.data, self.labels = [], []
        with open(path, 'r') as f:
            for line in f.readlines():
                result = json.loads(line)
                self.data.append(result["channel"])
                self.labels.append(result["label"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    def get_labels(self):
        return self.labels

def sc_collate_fn(batch_data):
    sent_seq = [x[0] for x in batch_data]
    labels = torch.tensor([x[1] for x in batch_data], dtype=torch.float32)
    padded_sent_seq = tokenizer(sent_seq, padding=True, truncation=True, max_length=2048, return_tensors="pt")
    data_length = torch.tensor([sum(mask) for mask in padded_sent_seq["attention_mask"]])
    return padded_sent_seq["input_ids"], padded_sent_seq["attention_mask"], data_length, labels


class MCDataset(Dataset):
    def __init__(self, path):
        self.labels = []
        self.c1, self.c2, self.c3, self.c4 = [], [], [], []
        with open(path, 'r') as f:
            cnt: int = 0
            for line in f:
                cnt += 1
                # if cnt > 2000:
                #     break
                if cnt % 5 == 1:
                    # 第一行: [索引 UniqueKey PID 标签]
                    line = line.split("\t")[-1]
                    self.labels.append(int(line, base=10))
                elif cnt % 5 == 2:
                    self.c1.append(line)
                elif cnt % 5 == 3:
                    self.c2.append(line)
                elif cnt % 5 == 4:
                    self.c3.append(line)
                elif cnt % 5 == 0:
                    self.c4.append(line)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.c1[idx], self.c2[idx], self.c3[idx], self.c4[idx], self.labels[idx]

def mc_collate_fn(batch_data):
    data_length = [len(x[0]) for x in batch_data]
    sent_seq_c1 = [x[0] for x in batch_data]
    sent_seq_c2 = [x[1] for x in batch_data]
    sent_seq_c3 = [x[2] for x in batch_data]
    sent_seq_c4 = [x[3] for x in batch_data]
    labels = torch.tensor([x[4] for x in batch_data], dtype=torch.float32)
    padded_sent_seq_c1 = tokenizer(sent_seq_c1, padding=True, truncation=True, max_length=4096, return_tensors="pt")
    padded_sent_seq_c2 = tokenizer(sent_seq_c2, padding=True, truncation=True, max_length=4096, return_tensors="pt")
    padded_sent_seq_c3 = tokenizer(sent_seq_c3, padding=True, truncation=True, max_length=4096, return_tensors="pt")
    padded_sent_seq_c4 = tokenizer(sent_seq_c4, padding=True, truncation=True, max_length=4096, return_tensors="pt")
    return padded_sent_seq_c1["input_ids"], padded_sent_seq_c2["input_ids"],padded_sent_seq_c3["input_ids"],padded_sent_seq_c4["input_ids"], data_length, labels

if __name__ == "__main__":
    # sent_seq = ["C:\\Windows\\system32\\rundll32.exe Windows.Storage.ApplicationData.dll,CleanupTemporaryState"]
    # padded_sent_seq = tokenizer(sent_seq, padding=True, truncation=True, max_length=2048, return_tensors="pt")
    output = tokenizer.encode("C:\\Windows\\system32\\rundll32.exe Windows.Storage.ApplicationData.dll,CleanupTemporaryState")
    toutput = tokenizer.decode(output)
    
    print(len(output), len(toutput))
    print(output)
    print(toutput)

    with open(config["pretrain"]["bert_pretrain_uncased"] + "vocab.txt", "r+") as f:
        vocab = [line.strip() for line in f.readlines()]

    ttoutput = []
    for i in output:
        ttoutput.append(vocab[i])
    print(ttoutput)