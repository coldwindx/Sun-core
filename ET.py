import argparse
import pickle
import random
import re

from loguru import logger
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from torch.utils.data import ConcatDataset, random_split
from dataset import ScDataset
from tools import Config, Notice
import metrics

seed = 6
random.seed(seed)
np.random.seed(seed)
CONFIG = Config()

def create_vocab(datasets):

    txt = [re.split(r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*', data) for data, _ in datasets]
    txt = [" ".join(line) for line in txt]

    tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, min_df=0.1, max_df=0.95)
    X = tv.fit_transform(txt)
    with open("./ckpt/ScETTask/TfidfVectorizer.ckpt", "wb") as f:
        pickle.dump(tv, f)
        
    n_samples, n_columns = X.shape
    print(f"X.shape is ({n_samples}, {n_columns})!")
    print(tv.get_feature_names_out())

def training(train_dataset, validate_dataset):
    clf = ExtraTreesClassifier(n_estimators=64, min_samples_leaf=128, max_depth=128, n_jobs=16, class_weight="balanced")
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
    # txt = [re.split(r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*', test_datasets.dataset[i][0]) for i in test_datasets.indices]
    # txt = [re.split(r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*', line) for ds in test_dataset for line in ds.data]
    # test_txt = [re.split(r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*', line) for line in test_dataset.data]
    test_txt = [" ".join(line) for line in test_txt]
    X_test = tv.transform(test_txt)
    with open("./ckpt/ScETTask/ExtraTreesClassifier.ckpt", "rb") as f:
        clf = pickle.load(f)
    predictions = clf.predict(X_test)

    y_hat = [1 if i >= 0.5 else 0 for i in predictions]
    labels = [label for _, label in test_datasets]
    
    tn, fp, fn, tp = confusion_matrix(labels, y_hat).ravel()
    acc = metrics.accurary(tp, tn, fp, fn)
    pre = metrics.precision(tp, tn, fp, fn)
    rec = metrics.recall(tp, tn, fp, fn)
    fprv = metrics.fpr(tp, tn, fp, fn)
    auc = 2 * pre * rec / (pre + rec)
    print(f"tp: {tp}\ntn: {tn}\nfp: {fp}\nfn: {fn}\nacc: {acc}\npre: {pre}\nrec: {rec}\nfpr: {fprv}\nauc: {auc}")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', default='train', type=str)
        args = parser.parse_args()

        train_dataset = ScDataset(CONFIG["datasets"]["train"])
        validate_dataset = ScDataset(CONFIG["datasets"]["validate"])
        # test_dataset = ScDataset(CONFIG["datasets"]["test"])
        # trainz_dataset = ScDataset(CONFIG["datasets"]["trainz"])
        # testz_dataset = ScDataset(CONFIG["datasets"]["testz"])
        full_dataset = ConcatDataset([train_dataset, validate_dataset])

        if args.mode == "tfidf":
            create_vocab(full_dataset)
        
        train_size = int(0.6 * len(full_dataset))
        val_size = int(0.2 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, test_size])
        if args.mode == "train":
            training(train_dataset, val_dataset)
        # if args.mode == "test":
            # test_dataset = ConcatDataset([ScDataset(CONFIG["datasets"]["test"]), ScDataset(CONFIG["datasets"]["testz"])])
            test_dataset = ScDataset(CONFIG["datasets"]["test"])
            testing(test_dataset)
    except Exception as e:
        logger.exception(e)
    Notice().send("[+] Training finished!")
