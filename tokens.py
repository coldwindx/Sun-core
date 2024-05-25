from itertools import chain
import re
from tokenizers import Tokenizer
from tokenizers.models import WordPiece

from dataset import ScDataset
from torch.utils.data import ConcatDataset
from transformer import CONFIG

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

# Normalizers进行标准化、小写，并将音标转换成字母
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

# Pre-tokenizers使用按空格分割（同时分割标点）
from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()

# Post-Processors处理成BERT的句子格式
from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
)

# 加载数据集
train_dataset = ScDataset(CONFIG["datasets"]["train"])
validate_dataset = ScDataset(CONFIG["datasets"]["validate"])
trainz_dataset = ScDataset(CONFIG["datasets"]["trainz"])
full_dataset = ConcatDataset([train_dataset, validate_dataset])

# 清洗数据
txts = [re.split(r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*', data) for data, _ in full_dataset]
numeric_pattern = re.compile(r'^[#\d]+$')
md5_pattern = re.compile(r'\b[a-fA-F0-9]{32}\b') 
txts = [word for word in list(chain.from_iterable(txts)) if not md5_pattern.match(word) and not numeric_pattern.match(word)]

# 加载训练器开始训练
from tokenizers.trainers import WordPieceTrainer
trainer = WordPieceTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train_from_iterator(txts, trainer)
tokenizer.save("token.json")

print("tokens.py run over!")

# import json
# import re

# from tools import Config

# with open("token.json", "r+") as f:
#     tokens = json.load(f)
# vocabs = list(tokens["model"]["vocab"].keys())

# # 排除仅包括数字的词
# numeric_pattern = re.compile(r'^[#\d]+$')
# vocabs = [word for word in vocabs if not numeric_pattern.match(word)]
# # 排除疑似MD5哈希值
# md5_pattern = re.compile(r'\b[a-fA-F0-9]{32}\b') 
# vocabs = [word for word in vocabs if not md5_pattern.match(word)]
# # 排除在bert_pretrain_uncased的词
# config = Config()
# with open(config["pretrain"]["bert_pretrain_uncased"] + "vocab.txt", "r+") as f:
#     vocab_bert = [line.strip() for line in f.readlines()]
# vocabs = [word for word in vocabs if word not in vocab_bert]

# for word in vocabs[:993]:
#     print(word)