import json
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer
from tokenizers.models import WordPiece

from dataset import ScDataset
from transformer import CONFIG

def create_tokenizer():
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

    # 加载训练器开始训练
    from tokenizers.trainers import WordPieceTrainer
    trainer = WordPieceTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(txts, trainer)
    tokenizer.save("token.json")

    print("tokens.py run over!")

def create_vocab():
    # dataset = ScDataset(CONFIG["datasets"]["train"])
    # txt = [re.split(r'[\\_/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*', data.lower()) for data, _ in dataset]
    # txt = [" ".join(line) for line in txt]

    # tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, max_features=35535 * 16)
    # x = tv.fit_transform(txt)

    # df = pd.DataFrame({'word': tv.get_feature_names_out(), 'tfidf': x.sum(axis=0).tolist()[0]})
    # df = df[~df['word'].str.contains(r'\d')]
    # df = df.sort_values(by="tfidf", ascending=False, inplace=True)
    # df.to_csv('tfidf.csv')

    df = pd.read_csv('tfidf.csv')
    df = df[~df['word'].str.match(r'^(.)\1+$').fillna(False)]
    df.sort_values(by="tfidf", ascending=False, inplace=True)
    print(df[:1024])
    print(len(df))

    with open("vocab.txt", "r") as f:
        vocab = [word.strip() for word in f.readlines()]
    nvocab = []
    c: int = 0
    for row in df.itertuples():
        if row.word not in vocab:
            if f"[unused{c}]" not in vocab:
                break
            for i, v in enumerate(vocab):
                if v == f"[unused{c}]":
                    nvocab.append(row.word)
                    vocab[i] = row.word
                    c += 1
                    break
    json.dump(nvocab, open("vocab_additional.json", "w"))
    with open("nvocab.txt", "w+") as f:
        f.writelines([word + "\n" for word in vocab])

def replace_vocab():
    with open("vocab.txt", "r") as f:
        vocab = [word.strip() for word in f.readlines()]
    tokens = json.load(open("token.json", 'r'))
    tokens = tokens["model"]["vocab"]
    nvocab = []
    c: int = 0
    for key, _ in tokens.items():
        if any(char.isdigit() for char in key):
            continue
        if key not in vocab:
            if f"[unused{c}]" not in vocab:
                break
            for i, v in enumerate(vocab):
                if v == f"[unused{c}]":
                    nvocab.append(key)
                    c += 1
                    break
    json.dump(nvocab, open("vocab_additional.json", "w"))

if __name__ == "__main__":
    create_vocab()
    # replace_vocab()