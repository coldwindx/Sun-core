import argparse
import os
import pickle
import random
import re

from loguru import logger
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import random_split
from dataset import ScDataset
from tools import Config, Notice
from torcheval.metrics.functional import binary_confusion_matrix, binary_accuracy, \
            binary_precision, binary_recall, binary_f1_score, binary_auroc

random.seed(42)
np.random.seed(42)
CONFIG = Config()


with open("./ckpt/ScETTask/TfidfVectorizer.ckpt", "rb") as f:
    tv = pickle.load(f)

full_dataset = ScDataset(CONFIG["datasets"]["train"])
test_dataset = ScDataset(CONFIG["datasets"]["test"])

train_size = int(0.6 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, test_size])
test_txt = [re.split(r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*', data) for data, _ in test_datasets]

test_txt = [" ".join(line) for line in test_txt]
X_test = tv.transform(test_txt)
