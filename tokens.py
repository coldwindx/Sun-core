import re
from tokenizers import Tokenizer
from tokenizers.models import WordPiece

from dataset import ScDataset
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

# 清洗数据
def process_string(s):
    s = re.sub(r'[^0-9a-zA-Z]', ' ', s)
    s = re.sub(r'(?<!\s)([A-Z])', r' \1', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s
# txts = [process_string(data) for data, _ in train_dataset]
numeric_pattern = re.compile(r'^[#\d]+$')
md5_pattern = re.compile(r'\b[a-fA-F0-9]{32}\b')
txts = []
for data, _ in train_dataset:
    s = process_string(data)
    s = [word for word in s.split(" ") if not md5_pattern.match(word) and not numeric_pattern.match(word)]
    txts.append(" ".join(s))

# counter = collections.defaultdict(int)
# txts = [word for word in list(chain.from_iterable(txts)) if not md5_pattern.match(word) and not numeric_pattern.match(word)]

# 加载训练器开始训练
from tokenizers.trainers import WordPieceTrainer
trainer = WordPieceTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train_from_iterator(txts, trainer)
tokenizer.save("token.json")

print("tokens.py run over!")
