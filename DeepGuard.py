import argparse
import os
import random
from loguru import logger
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import torch.nn.functional as F
import lightning as pl

from tools import Config, Notice
import metrics

seed = 6
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)                    # 为CPU设置随机种子
torch.cuda.manual_seed(seed)               # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)           # 为所有GPU设置随机种子
pl.seed_everything(seed, workers=True)
torch.set_float32_matmul_precision(precision="high")
CONFIG = Config()

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
    def forward(self, x):
        z = self.encoder(x)
        h = self.decoder(z)
        return z, h

class DeepGuard(pl.LightningModule):
    def __init__(self, input_dim, model_dim, output_dim, lr, warmup, max_iters, dropout=0.0, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.batchlayer = nn.BatchNorm1d(num_features=input_dim)
        self.autoencoder = AutoEncoder(self.hparams.input_dim, self.hparams.model_dim, self.hparams.output_dim, self.hparams.dropout)
    def forward(self, x):
        x = self.batchlayer(x)
        z, h = self.autoencoder(x)
        return x, h
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    def test_step(self, batch, batch_idx):
        raise NotImplementedError


class DeepGuardPredictor(DeepGuard):
    def _calculate_loss(self, batch, mode="train"):
        x, _ = batch
        x, h = self.forward(x)
        # loss = F.cross_entropy(x, h)
        loss = F.mse_loss(x, h)
        self.log("%s_loss" % mode, loss, on_epoch=True, enable_graph=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="val")
        return loss
    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="test")
        return loss
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        x, h = self.forward(x)
        # loss = F.cross_entropy(x, h)
        loss = F.mse_loss(x, h, reduce=False).mean(dim=1)
        return loss

class DeepGuardDataset(Dataset):
    def __init__(self, mode = "train"):
        dataset = np.load(CONFIG["datasets"]["deep_guard_dataset"])
        if mode == "train":
            self.data = torch.from_numpy(dataset["X_train"]).float()
            self.labels = torch.from_numpy(dataset["Y_train"]).float()
            self.data = torch.nn.functional.normalize(self.data, dim=0)

        if mode == "test":
            self.data = torch.from_numpy(dataset["X_test"]).float()
            self.labels = torch.from_numpy(dataset["Y_test"]).float()
            self.data = torch.nn.functional.normalize(self.data, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    def get_labels(self):
        return self.labels

def training(train_dataset, val_dataset, args, **kwargs):
    root_dir = os.path.join(CONFIG["checkpoint"]["path"], "DeepGuardTask")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(every_n_epochs=1, save_top_k=-1)],
        accelerator="auto",
        devices=1,
        max_epochs=50,
        # accumulate_grad_batches=4,
        # limit_train_batches= 5000, 
        # enable_progress_bar=False
    )
    trainer.logger._default_hp_metric = None
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = DeepGuardPredictor(max_iters=trainer.max_epochs * len(train_loader), **kwargs)
    if args.enable_ckpt:
        pretrained_filename = CONFIG["checkpoint"]["path"] + "DeepGuardTask/lightning_logs" + CONFIG["checkpoint"]["DeepGuardTask"]
        trainer.fit(model, train_loader, val_loader, ckpt_path=pretrained_filename)
    else:
        trainer.fit(model, train_loader, val_loader)
    return model

def testing(test_dataset, **kwargs):
    # load model
    pretrained_filename = CONFIG["checkpoint"]["path"] + "DeepGuardTask/lightning_logs" + CONFIG["checkpoint"]["DeepGuardTask"]

    trainer = pl.Trainer(enable_checkpointing=False, logger=False)
    classifier = DeepGuardPredictor.load_from_checkpoint(pretrained_filename)
    classifier.eval()

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    labels = DeepGuardDataset("test").labels.numpy()
    scores = trainer.predict(classifier, dataloaders=test_loader)
    scores = torch.cat(scores, dim=0).numpy()

    print(len(scores), len(labels))

    from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                                precision_score, recall_score, roc_curve)

    def plot_roc(labels, scores):
        fpr, tpr, thresholds = roc_curve(labels, scores)
        maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
        threshold = thresholds[maxindex]
        print('异常阈值', threshold)
        auc_score = auc(fpr, tpr)
        print('auc值: {:.4f}'.format(auc_score))
        return threshold, auc_score


    def eval(labels, pred):
        plot_roc(labels, pred)
        print(confusion_matrix(labels, pred))
        a, b, c, d = accuracy_score(labels, pred), precision_score(
            labels, pred), recall_score(labels, pred), f1_score(labels, pred)
        print("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(a, b, c, d))
        return a, b, c, d


    def matrix(true_graph_labels, scores):
        t, auc = plot_roc(true_graph_labels, scores)
        true_graph_labels = np.array(true_graph_labels)
        scores = np.array(scores)
        pred = np.ones(len(scores))
        pred[scores < t] = 0
        print(confusion_matrix(true_graph_labels, pred))
        print("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(accuracy_score(true_graph_labels, pred), precision_score(
            true_graph_labels, pred), recall_score(true_graph_labels, pred), f1_score(true_graph_labels, pred)))
        return auc, precision_score(true_graph_labels, pred), recall_score(true_graph_labels, pred), f1_score(true_graph_labels, pred)

    pred = torch.zeros(len(scores))
    idx = scores.argsort()[::-1]  # 从大到小

    # for k in [500, 2000, 4000, 6000, 8000, 10000]:
    #     print('============ k=', k)
    #     nidx = np.ascontiguousarray(idx[:k])
    #     pred[np.sort(nidx)] = 1  # 异常分数最高的K为样本判定为异常
    #     a, b, c, d = eval(labels.astype(np.longfloat), pred)
    #     print("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(a, b, c, d))

    y_hat = [1 if i >= scores[idx[4000]] else 0 for i in scores]

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
        parser.add_argument('--enable_ckpt', action="store_true", help="Run with ckpt")
        args = parser.parse_args()

        train_dataset = DeepGuardDataset("train")
        test_dataset = DeepGuardDataset("test")

        train_size = int(0.6 * len(train_dataset))
        val_size = int(0.2 * len(train_dataset))
        test_size = len(train_dataset) - train_size - val_size
        train_dataset, val_dataset, _ = random_split(train_dataset, [train_size, val_size, test_size])

        if args.mode == "train":
            model = training(
                train_dataset, 
                val_dataset,
                args,
                input_dim=22,
                model_dim=16,
                output_dim=8,
                dropout=0.2,
                lr=1e-4,
                warmup=50,
                weight_decay=1e-6
            )
        if args.mode == "test":
            testing(test_dataset)
    except Exception as e:
        logger.exception(e)
    Notice().send("[+] Training finished!")