import argparse
import numpy as np
import datatable as dt
from loguru import logger
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
import torch
import torch.nn.functional as F
from torcheval.metrics.functional import *
import lightning as pl

from tools import Config

CONFIG = Config()
pl.seed_everything(42, workers=True)
torch.set_float32_matmul_precision(precision="high")

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

# class DeepGuardDataset(Dataset):
#     def __init__(self, path):
#         dataset = np.load(path)
#         self.datas = dataset["X"].astype(np.float32)
#         self.labels = dataset["label"].astype(np.float32)
#     def __len__(self):
#         return len(self.datas)

#     def __getitem__(self, idx):
#         return self.datas[idx], self.labels[idx]
#     def get_labels(self):
#         return self.labels

class DeepGuardDataset(Dataset):
    def __init__(self, path):
        dataset = dt.fread(path, fill=True).to_pandas()
        events = ["ProcessStart", "ProcessEnd", "ThreadStart", "ThreadEnd", "ImageLoad", "FileIOWrite", "FileIORead", "FileIOFileCreate", "FileIORename", "FileIOCreate", "FileIOCleanup", "FileIOClose", "FileIODelete", "FileIOFileDelete", "RegistryCreate", "RegistrySetValue", "RegistryOpen", "RegistryDelete", "RegistrySetInformation", "RegistryQuery", "RegistryQueryValue", "CallStack"]
        x = [dataset["channel"].apply(lambda x:x.count(event)).to_numpy() for event in events]
        self.datas = np.vstack(x).T.astype(np.float32)
        self.labels = dataset["label"].to_numpy()
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx], self.labels[idx]
    def get_labels(self):
        return self.labels

def training(train_dataset, val_dataset, args, **kwargs):

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    trainer = pl.Trainer(
        callbacks=[StochasticWeightAveraging(swa_lrs=1e-2), ModelCheckpoint(every_n_epochs=1, save_top_k=-1)],
        # accelerator="gpu", devices=1, num_nodes=1, strategy="ddp",
        accelerator="auto",
        max_epochs=100,
        accumulate_grad_batches=8
    )
    trainer.logger._default_hp_metric = None
    ### start to train
    model = DeepGuardPredictor(max_iters=trainer.max_epochs * len(train_loader), **kwargs)
    trainer.fit(model, train_loader, val_loader)

def testing(predictor, test_dataset, output, **kwargs):
    # trainer = pl.Trainer(enable_checkpointing=False, logger=False, devices="auto")
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # scores = trainer.predict(predictor, dataloaders=test_loader)
    # scores = torch.cat(scores, dim=0).cuda()
    # np.save(open(output, "wb"), scores.cpu().numpy())
    scores = torch.from_numpy(np.load(output)).cuda()
    labels = torch.from_numpy(test_dataset.labels).flatten().cuda().int()
  
    f1s, thress, idx = [], [], scores.argsort(descending=True)
    for k in range(100, len(scores), 100):
        thress.append(scores[idx[k]])
        # f1s.append(binary_f1_score(scores, labels, threshold = scores[idx[k]]).item())
        precision = binary_precision(scores, labels, threshold = scores[idx[k]]).item()
        recall = binary_recall(scores, labels, threshold = scores[idx[k]]).item()
        f1s.append(abs(precision - recall))
    
    f1sid = np.argsort(np.array(f1s))
    thres = thress[f1sid[5]]
    print(thres)

    cm = binary_confusion_matrix(scores, labels, threshold = thres)
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    accuracy = binary_accuracy(scores, labels, threshold = thres)
    precision = binary_precision(scores, labels, threshold = thres).item()
    recall = binary_recall(scores, labels, threshold = thres).item()
    f1 = binary_f1_score(scores, labels, threshold = thres).item()
    auc = binary_auroc(scores, labels).item()
    logger.info(f"\ntp: {tp}\ntn: {tn}\nfp: {fp}\nfn: {fn}\nacc: {accuracy}\npre: {precision}\nrec: {recall}\nauc: {auc}\nf1: {f1}")


parser = argparse.ArgumentParser()
parser.add_argument('--task', default='eval', type=str)
# parser.add_argument('--path', default='/mnt/sdd1/data/zhulin/jack/datasets/deepguard.train.npz', type=str)
parser.add_argument('--path', default='/mnt/sdd1/data/zhulin/jack/cdatasets.test.5.csv', type=str)
parser.add_argument('--model', default='/mnt/sdd1/data/zhulin/jack/models/DeepGuard-100.ckpt', type=str)
parser.add_argument('--output', default='/mnt/sdd1/data/zhulin/jack/scores/DeepGuardPredictor.npy', type=str)
parser.add_argument('--enable_ckpt', action="store_true", help="Run with ckpt")

args = parser.parse_args()

if args.task == "train":
    dataset = DeepGuardDataset(args.path)
    train_dataset, val_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    
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

if args.task == "eval":
    dataset = DeepGuardDataset(args.path)
    predictor = DeepGuardPredictor.load_from_checkpoint(args.model)
    predictor.eval()
    testing(predictor, dataset, args.output)