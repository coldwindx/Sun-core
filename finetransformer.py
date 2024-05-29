import os

from dataset import ScDataset, sc_collate_fn
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# from datasets import load_dataset

# raw_datasets = load_dataset("imdb")

#分词器（用的bert基线模型）
from transformers import AutoTokenizer
from tools import Config
BERT_PATH = '/home/zhulin/pretrain/bert-tiny/'
from transformers import BertModel,BertTokenizer

tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
bert = BertModel.from_pretrained(BERT_PATH)

# tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
# #定义模型
# from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH, num_labels=2)

# #文本截断：批量处理
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)

# tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# #获取数据集中的一部分，进行训练（非必须，主要是快）
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000)) 
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000)) 
# full_train_dataset = tokenized_datasets["train"]
# full_eval_dataset = tokenized_datasets["test"]


# #定义数据加载器，使用它来进行迭代批次。tokenized_datasets在执行此操作之前，需要对其进行一些后处理：

# #删除与模型不期望的值相对应的列（这里是"text"列）
# #将列重命名"label"为"labels"（因为模型期望参数被命名labels）
# #设置数据集的格式，以便它们返回 PyTorch 张量而不是列表。
# tokenized_datasets = tokenized_datasets.remove_columns(["text"])
# tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# tokenized_datasets.set_format("torch")

# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

#定义数据加载器
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch.utils.data import DataLoader

# train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
# eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)


CONFIG = Config()
train_dataset = ScDataset(CONFIG["datasets"]["train"])
trainz_dataset = ScDataset(CONFIG["datasets"]["trainz"])
validate_dataset = ScDataset(CONFIG["datasets"]["validate"])
test_dataset = ScDataset(CONFIG["datasets"]["test"])
testz_dataset = ScDataset(CONFIG["datasets"]["testz"])
        
full_dataset = ConcatDataset([train_dataset, validate_dataset, trainz_dataset])
test_dataset = ConcatDataset([test_dataset, testz_dataset])

train_size = int(0.6 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, test_size])

def sc_bert_fn(batch_data):
    padded_sent_seq = tokenizer([x[0] for x in batch_data], padding=True, truncation=True, max_length=512, return_tensors="pt")
    labels = torch.tensor([[x[1], 1 - x[1]] for x in batch_data], dtype=torch.float32)
    return padded_sent_seq, labels

train_loader = DataLoader(train_dataset, batch_size=256, collate_fn=sc_bert_fn, num_workers=0)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=sc_collate_fn, num_workers=4)

#优化器和学习率调度器
from transformers import AdamW

optimizer = AdamW(bert.parameters(), lr=5e-5)

#学习率设置为从最大值（此处为 5e-5）到 0 的线性衰减
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

#定义一个device放置模型（如果有GPU可用，会快很多）
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
bert.to(device)

#开始训练，为直观展示，训练步骤中添加了一个进度条
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

bert.train()
for epoch in range(num_epochs):
    for batch, labels in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = bert(**batch)
        
        # pooler_output = outputs.pooler_output

        # loss = outputs.loss

        # loss.backward()

        # optimizer.step()
        # lr_scheduler.step()
        # optimizer.zero_grad()
        progress_bar.update(1)

# #验证评估
# metric= load_metric("accuracy")
# model.eval()
# for batch in eval_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch)

#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     metric.add_batch(predictions=predictions, references=batch["labels"])

# metric.compute()
