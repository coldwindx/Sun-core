import argparse
import os
import pickle
import random
import re

from loguru import logger
import numpy as np
import datatable as dt

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

def create_vocab(datasets):

    txt = [re.split(r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*', data) for data, _ in datasets]
    txt = [" ".join(line) for line in txt]

    tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, min_df=0.1, max_df=0.95)
    X = tv.fit_transform(txt)
    os.makedirs("./ckpt/ScETTask/", exist_ok=True)
    with open("./ckpt/ScETTask/TfidfVectorizer.ckpt", "wb") as f:
        pickle.dump(tv, f)
        
    n_samples, n_columns = X.shape
    print(f"X.shape is ({n_samples}, {n_columns})!")
    # print(tv.get_feature_names_out())

def training(train_dataset, validate_dataset):
    clf = ExtraTreesClassifier(n_estimators=64, min_samples_leaf=2048, max_depth=64, n_jobs=16, class_weight="balanced")
    with open("./ckpt/ScETTask/TfidfVectorizer.ckpt", "rb") as f:
        tv: TfidfVectorizer = pickle.load(f)
    train_txt = [re.split(r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*', train_dataset.dataset[i][0]) for i in train_dataset.indices]
    train_txt = [" ".join(line) for line in train_txt]
    X_train = tv.transform(train_txt)
    Y_train = [train_dataset.dataset[i][1] for i in train_dataset.indices]
    clf.fit(X_train, Y_train)
    with open("./ckpt/ScETTask/ExtraTreesClassifier.ckpt", "wb") as f:
        pickle.dump(clf, f)

def testing(test_datasets):
    with open("./ckpt/ScETTask/TfidfVectorizer.ckpt", "rb") as f:
        tv = pickle.load(f)
    test_txt = [re.split(r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*', data) for data, _ in test_datasets]
    test_txt = [" ".join(line) for line in test_txt]
    X_test = tv.transform(test_txt)
    with open("./ckpt/ScETTask/ExtraTreesClassifier.ckpt", "rb") as f:
        clf = pickle.load(f)
    predictions = clf.predict(X_test)

    ### save scores
    np.save(open("/mnt/sdd1/data/zhulin/jack/scores/ExtraTreesClassifier.npy", "wb"), predictions)
    ### binary_confusion_matrix
    scores = torch.from_numpy(predictions).cuda()
    labels = torch.tensor([label for _, label in test_datasets], device="cuda")
    cm = binary_confusion_matrix(scores, labels)
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    accuracy = binary_accuracy(scores, labels, threshold = 0.5)
    precision = binary_precision(scores, labels, threshold = 0.5).item()
    recall = binary_recall(scores, labels, threshold = 0.5).item()
    f1 = binary_f1_score(scores, labels, threshold = 0.5).item()
    auc = binary_auroc(scores, labels).item()

    print(f"tp: {tp}\ntn: {tn}\nfp: {fp}\nfn: {fn}\nacc: {accuracy}\npre: {precision}\nrec: {recall}\nauc: {auc}\nf1: {f1}")


parser = argparse.ArgumentParser()
parser.add_argument('--task', default='train', type=str)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

pattern = r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*'

if args.task == "eval":
    # load model
    tv = pickle.load(open("./ckpt/ScETTask/TfidfVectorizer.ckpt", "rb"))
    clf = pickle.load(open("./ckpt/ScETTask/ExtraTreesClassifier.ckpt", "rb"))
    # load dataset
    dataset = dt.fread(args.dataset, fill=True).to_pandas()
    # dataset["channel"] = dataset["channel"].apply(lambda txt: " ".join(re.split(pattern, txt)))
    # X = dataset["channel"].to_numpy()

    # scores = clf.predict(tv.transform(X))
    # np.save(open("/mnt/sdd1/data/zhulin/jack/scores/ExtraTreesClassifier.npy", "wb"), scores)
    scores = np.load("/mnt/sdd1/data/zhulin/jack/scores/ExtraTreesClassifier.npy", allow_pickle=True)
    ### binary_confusion_matrix
    scores = torch.from_numpy(scores).cuda()
    labels = torch.from_numpy(dataset["label"].to_numpy(dtype=np.int32)).cuda()

    cm = binary_confusion_matrix(scores, labels)
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    accuracy = binary_accuracy(scores, labels, threshold = 0.5)
    precision = binary_precision(scores, labels, threshold = 0.5).item()
    recall = binary_recall(scores, labels, threshold = 0.5).item()
    f1 = binary_f1_score(scores, labels, threshold = 0.5).item()
    auc = binary_auroc(scores, labels).item()

    print(f"tp: {tp}\ntn: {tn}\nfp: {fp}\nfn: {fn}\nacc: {accuracy}\npre: {precision}\nrec: {recall}\nauc: {auc}\nf1: {f1}")