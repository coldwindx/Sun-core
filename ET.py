

import argparse
import random
import re

from loguru import logger
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace, Split
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from torch.utils.data import ConcatDataset, DataLoader, random_split
from sklearn.feature_extraction.text import TfidfVectorizer
from dataset import ScDataset
from tools import Config

seed = 6
random.seed(seed)
np.random.seed(seed)
CONFIG = Config()

def create_vocab(datasets):
    txt = datasets.data[:200]
    txt = [re.split(r'[\\/=:,.\s{}]\s*', line) for line in txt]
    txt = [" ".join(line) for line in txt]
    print(txt)
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )
    trainer = WordPieceTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(txt, trainer)
    tokenizer.save("token.json")
    # tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
    # tv.fit_transform(txt)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', default='train', type=str)
        args = parser.parse_args()

        # train_dataset = ScDataset(CONFIG["datasets"]["train"])
        validate_dataset = ScDataset(CONFIG["datasets"]["validate"])
        # test_dataset = ScDataset(CONFIG["datasets"]["test"])
        
        create_vocab(validate_dataset)


        # full_dataset = ConcatDataset([train_dataset, validate_dataset])
        # train_size = int(0.6 * len(full_dataset))
        # val_size = int(0.2 * len(full_dataset))
        # test_size = len(full_dataset) - train_size - val_size
        # train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    except Exception as e:
        logger.exception(e)
    # Notice().send("[+] Training finished!")
