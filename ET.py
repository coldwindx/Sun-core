import argparse
import pickle
import random
import re

from loguru import logger
import numpy as np

from scipy import sparse
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

def create_vocab(datasets, *, n_class = 2, tfidf = False, sw=False, sb=False):
    if tfidf:
        txt = [re.split(r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*', data) for data, _ in datasets]
        # txt = [re.split(r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*', line) for ds in datasets for line in ds.data]
        txt = [" ".join(line) for line in txt]

        tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, 
                             min_df=0.1, max_df=0.95)
        X = tv.fit_transform(txt)
        with open("TfidfVectorizer.ckpt", "wb") as f:
            pickle.dump(tv, f)
        sparse.save_npz("tfidf", X)
    else:
        print("Loading tfidf.npz!")
        X = sparse.load_npz("tfidf.npz")
        print("Load tfidf.npz over!")

    n_samples, n_columns = X.shape
    print(f"X.shape is ({n_samples}, {n_columns})!")
    print(tv.get_feature_names_out())
    return

    y = []
    for ds in datasets: y.extend(ds.get_labels())
    y = np.array(y)
    # step-1. 计算每类数据的d维均值向量
    mean_vectors = []
    for cls in range(0, n_class):
        mean_vectors.append(X[y == cls].mean(axis=0).A[0])
        print('Mean Vector class %s: %s\n' %(cls, mean_vectors[cls]))

    # step-2. 计算散布矩阵 [*]
    S_W = np.zeros((n_columns, n_columns))
    if sw:
        for cl, mv in zip(range(0, n_class), mean_vectors):
            class_sc_mat = np.zeros((n_columns, n_columns))                                 # scatter matrix for every class
            for row in X[y == cl]:                                                        
                row, mv = row.toarray().reshape(n_columns,1), mv.reshape(n_columns,1)       # make column vectors
                class_sc_mat += (row-mv).dot((row-mv).T)
            S_W += class_sc_mat                                                             # sum class scatter matrices
        print('within-class Scatter Matrix:\n', S_W)
        np.save("S_W", S_W)
    else:
        S_W = np.load("S_W.npy")
    overall_mean = np.mean(X, axis=0)

    S_B = np.zeros((n_columns, n_columns))
    if sb:
        for cls,mean_vec in enumerate(mean_vectors):  
            n = X[y==cls,:].shape[0]
            mean_vec = mean_vec.reshape(n_columns,1)                            # make column vector
            overall_mean = overall_mean.reshape(n_columns,1)                    # make column vector
            S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
        print('between-class Scatter Matrix:\n', S_B)
        np.save("S_B", S_B)
    else:
        S_B = np.load("S_B.npy")

    # step-3. 求解矩阵 S−1WSB 的广义本征值
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:,i].reshape(n_columns,1)   
        print('\nEigenvector {}: \n{}'.format(i, eigvec_sc.real))
        print('Eigenvalue {:}: {:.2e}'.format(i, eig_vals[i].real))
    # step-4. 选择线性判别“器”构成新的特征子空间
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    print('Eigenvalues in decreasing order:\n')
    for i in eig_pairs:
        print(i[0])
    print('Variance explained:\n')
    eigv_sum = sum(eig_vals)
    for i,j in enumerate(eig_pairs):
        print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
    W = np.hstack((eig_pairs[0][1].reshape(n_columns,1), eig_pairs[1][1].reshape(n_columns,1)))
    print('Matrix W:\n', W.real)
    # step-5. 将样本变换到新的子空间中
    X_lda = X.dot(W)
    print(X_lda)
    # lda = LinearDiscriminantAnalysis(solver="svd", n_components=10)
    # Y = lda.fit_transform(X.toarray(), y)
    # print(X)

    # X = lda.transform(X.toarray())
    # print(Y.shape)

def training(train_dataset, validate_dataset):
    clf = ExtraTreesClassifier(n_estimators=64, min_samples_leaf=128, max_depth=128, n_jobs=16, 
                               class_weight={0: 1, 1: 10})
    with open("TfidfVectorizer.ckpt", "rb") as f:
        tv = pickle.load(f)
    train_txt = [re.split(r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*', train_dataset.dataset[i][0]) for i in train_dataset.indices]
    train_txt = [" ".join(line) for line in train_txt]
    X_train = tv.transform(train_txt)
    Y_train = [train_dataset.dataset[i][1] for i in train_dataset.indices]
    clf.fit(X_train, Y_train)
    with open("ExtraTreesClassifier.ckpt", "wb") as f:
        pickle.dump(clf, f)

def testing(test_datasets):
    with open("TfidfVectorizer.ckpt", "rb") as f:
        tv = pickle.load(f)
    txt = [re.split(r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*', test_datasets.dataset[i][0]) for i in test_datasets.indices]
    # txt = [re.split(r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*', line) for ds in test_dataset for line in ds.data]
    # test_txt = [re.split(r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*', line) for line in test_dataset.data]
    test_txt = [" ".join(line) for line in test_txt]
    X_test = tv.transform(test_txt)
    with open("ExtraTreesClassifier.ckpt", "rb") as f:
        clf = pickle.load(f)
    predictions = clf.predict(X_test)

    y_hat = [1 if i >= 0.5 else 0 for i in predictions]
    labels = [test_datasets.dataset[i][1] for i in test_datasets.indices]
    
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
        trainz_dataset = ScDataset(CONFIG["datasets"]["train"])
        # testz_dataset = ScDataset(CONFIG["datasets"]["test"])
        full_dataset = ConcatDataset([train_dataset, validate_dataset, trainz_dataset])

        if args.mode == "tfidf":
            create_vocab(full_dataset, tfidf=True)
        
        train_size = int(0.6 * len(full_dataset))
        val_size = int(0.2 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, test_size])
        if args.mode == "train":
            training(train_dataset, val_dataset)
        if args.mode == "test":
            test_dataset = ConcatDataset([ScDataset(CONFIG["datasets"]["test"]), ScDataset(CONFIG["datasets"]["test"])])
            testing(test_dataset)
    except Exception as e:
        logger.exception(e)
    Notice().send("[+] Training finished!")
