import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from tools import Config
from torcheval.metrics.functional import binary_confusion_matrix, binary_accuracy, \
            binary_precision, binary_recall, binary_f1_score, binary_auroc, binary_precision_recall_curve

import lightning as pl
sys.path.append("/home/zhulin/workspace/Sun-core")  
from transformer import ScPredictor
from dataset import ScDataset, sc_collate_fn

pl.seed_everything(42)
torch.set_float32_matmul_precision(precision="high")

def compute_metrics(args):
    ### parse path
    model_dir = "/".join(args.model.split("/")[:-1])

    ### load dataset
    full_dataset = ScDataset(Config()["datasets"]["train"])
    val_size = 1024 * 64
    test_size = 1024 * 64
    train_size = len(full_dataset) - val_size - test_size
    _, _, dataset = random_split(full_dataset, [train_size, val_size, test_size])
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=sc_collate_fn, num_workers=4)

    ### load model
    trainer = pl.Trainer(enable_checkpointing=False, logger=False)
    classifier = ScPredictor.load_from_checkpoint(args.model)
    classifier.eval()

    ### predict scores
    scores = trainer.predict(classifier, dataloaders=dataloader)
    scores = torch.cat(scores, dim=0)
    np.save(f"{model_dir}/scores", scores)

    ### binary_confusion_matrix
    scores = torch.from_numpy(np.load(f"{model_dir}/scores.npy")).cuda()
    labels = torch.tensor([dataset.dataset[i][1] for i in dataset.indices], device="cuda")
    cm = binary_confusion_matrix(scores, labels)
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    accuracy = binary_accuracy(scores, labels, threshold = 0.5)
    precision = binary_precision(scores, labels, threshold = 0.5).item()
    recall = binary_recall(scores, labels, threshold = 0.5).item()
    f1 = binary_f1_score(scores, labels, threshold = 0.5).item()
    auc = binary_auroc(scores, labels).item()

    print(f"tp: {tp}\ntn: {tn}\nfp: {fp}\nfn: {fn}\nacc: {accuracy}\npre: {precision}\nrec: {recall}\nauc: {auc}\nf1: {f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='epoch=29-step=2999010.ckpt', type=str)
    parser.add_argument('--data', default='epoch=29-step=2999010.ckpt', type=str)
    args = parser.parse_args()
    
    ### parse path
    model_dir = "/".join(args.model.split("/")[:-1])

    ### load dataset
    full_dataset = ScDataset(Config()["datasets"]["train"])
    val_size = 1024 * 64
    test_size = 1024 * 64
    train_size = len(full_dataset) - val_size - test_size
    _, _, dataset = random_split(full_dataset, [train_size, val_size, test_size])
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=sc_collate_fn, num_workers=4)

    ### load model
    trainer = pl.Trainer(enable_checkpointing=False, logger=False)
    classifier = ScPredictor.load_from_checkpoint(args.model)
    classifier.eval()

    ### predict scores
    # scores = trainer.predict(classifier, dataloaders=dataloader)
    # scores = torch.cat(scores, dim=0)
    # np.save(f"{model_dir}/scores", scores)
    
    ### Bucket evaluation
    scores = torch.from_numpy(np.load(f"{model_dir}/scores.npy")).cuda()
    labels = torch.tensor([dataset.dataset[i][1] for i in dataset.indices], device="cuda")

    precision, recall, thresholds = binary_precision_recall_curve(scores, labels)
    df = pd.DataFrame({"precision": precision[:-1].detach().cpu().numpy(), 
                       "recall": recall[:-1].detach().cpu().numpy(), 
                       "thresholds": thresholds.detach().cpu().numpy()})
    df.to_csv("./tmp/curve.csv")

    dff = df[(df["precision"] > 0.98) & (df["recall"] > 0.85)]
    print(dff)


    # buckets = pd.DataFrame({"labels": [dataset.dataset[i][1] for i in dataset.indices], "scores": scores})
    # buckets['bins'] = pd.qcut(buckets['scores'], q=20, duplicates='drop')

    # for i in range(0, 20):
    #     df = buckets[buckets["bins"].isin([19 - x for x in range(i + 1)])]
    #     s = torch.from_numpy(df["scores"].to_numpy()).cuda()
    #     l = torch.from_numpy(df["labels"].to_numpy()).cuda()
    #     t = df["scores"].min()
    #     # import pdb
    #     # pdb.set_trace()
    #     accuracy = binary_accuracy(s, l, threshold = t)
    #     precision = binary_precision(s, l, threshold = t).item()
    #     recall = binary_recall(s, l, threshold = t).item()
    #     f1 = binary_f1_score(s, l, threshold = t).item()
    #     auc = binary_auroc(s, l).item()
    #     print(f"前{1 + i}桶：threshold:{t}, acc: {accuracy}，pre: {precision}，rec: {recall}，auc: {auc}，f1: {f1}")
    