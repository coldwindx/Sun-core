import json
import os
import sys
from loguru import logger
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__PATH__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(__PATH__)  

class SCDataset(Dataset):
    def __init__(self, path):
        self.data, self.labels = [], []
        with open(path, 'r') as f:
            cnt: int = 0
            for line in f:
                cnt += 1
                if cnt % 2 == 1:
                    # 第一行: [索引 UniqueKey PID 标签]
                    line = line.split("\t")[-1]
                    self.labels.append(int(line, base=10))
                    continue
                if cnt % 2 == 0:
                    self.data.append(line)
                    continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    def get_labels(self):
        return self.labels

class ScDataset(Dataset):
    def __init__(self, path):
        self.data, self.labels = [], []
        with open(path, 'r') as f:
            cnt: int = 0
            for line in f.readlines():
                result = json.loads(line)
                self.data.append(result["channel"])
                self.labels.append(result["label"])
                cnt += 1
                if cnt > 2000:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    def get_labels(self):
        return self.labels


PRETRAIN_PATH = "/home/zhulin/pretrain/"
tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_PATH + 'bert_pretrain_uncased/')
def sc_collate_fn(batch_data):
    sent_seq = [x[0] for x in batch_data]
    labels = torch.tensor([x[1] for x in batch_data], dtype=torch.float32)
    padded_sent_seq = tokenizer(sent_seq, padding=True, truncation=True, max_length=2048, return_tensors="pt")
    data_length = [sum(mask) for mask in padded_sent_seq["attention_mask"]]
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

def prepare_pack_padded_sequence(inputs_words, seq_lengths, descending=True):
    """
    for rnn model
    :param device:
    :param inputs_words:
    :param seq_lengths:
    :param descending:
    :return:
    """
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=descending)
    _, desorted_indices = torch.sort(indices, descending=False)
    sorted_inputs_words = inputs_words[indices]
    return sorted_inputs_words, sorted_seq_lengths, desorted_indices

if __name__ == "__main__":
    
    try:
        # datasets = KellectDataset("")
        
        train_dataset = SCDataset("/home/zhulin/datasets/cdatasets_train.txt")
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, collate_fn=sc_collate_fn)
        input_net = torch.nn.Embedding(30522, 16)
        for batch in train_loader:
            inp_data, _, lengths, labels = batch
            text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(inp_data, lengths)
            embedded = input_net(text)
            sorted_seq_lengths = sorted_seq_lengths.cpu()
            packed_embedded = pack_padded_sequence(embedded, sorted_seq_lengths, batch_first=True)
            break
    except Exception as e:
        logger.exception(e)
    # Notice().send("[+] Datasets generating finished!\n")