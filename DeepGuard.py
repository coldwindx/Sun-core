import argparse
import os
import random
from loguru import logger
import numpy as np
from sklearn.metrics import confusion_matrix
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import torch.nn.functional as F
import lightning as pl

from dataset import sc_collate_fn
from sampler import ImbalancedDatasetSampler
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
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        h = self.decoder(z)
        return z, h

class DeepGuard(pl.LightningModule):
    def __init__(self, vocab_size, input_dim, model_dim, output_dim, lr, warmup, max_iters, dropout=0.0, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.autoencoder = AutoEncoder(self.hparams.input_dim, self.hparams.model_dim, self.hparams.output_dim, self.hparams.dropout)
    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder
    
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
        _, y = self.forward(x)
        loss = F.cross_entropy(x, y)
        self.log("%s_loss" % mode, loss, on_epoch=True, enable_graph=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="val")
        return loss
    def test_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="test")
        return loss
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        _, y = self.forward(x)
        loss = F.cross_entropy(x, y)
        return loss
    
def training(train_dataset, val_dataset, args, **kwargs):
    root_dir = os.path.join(CONFIG["checkpoint"]["path"], "DeepGuardTask")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(every_n_epochs=1, save_top_k=-1)],
        accelerator="auto",
        devices=1,
        max_epochs=40,
        accumulate_grad_batches=4,
        limit_train_batches= 5000, 
        # enable_progress_bar=False
    )
    trainer.logger._default_hp_metric = None
    model = DeepGuardPredictor(max_iters=trainer.max_epochs * len(train_loader), **kwargs)

    sampler = ImbalancedDatasetSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=sc_collate_fn, num_workers=4, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=sc_collate_fn, num_workers=4)

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

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=sc_collate_fn, num_workers=4)
    predictions = trainer.predict(classifier, dataloaders=test_loader)
    predictions = torch.cat(predictions, dim=0)
    y_hat = [1 if i >= 0.5 else 0 for i in predictions]
    labels = test_dataset.get_labels()
    
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

        dataset = np.load(CONFIG["datasets"]["deep_guard_dataset"])
        train_dataset = torch.from_numpy(dataset["X_train"])
        test_dataset = torch.from_numpy(dataset["X_test"])

        train_size = int(0.6 * len(train_dataset))
        val_size = int(0.2 * len(train_dataset))
        test_size = len(train_dataset) - train_size - val_size
        train_dataset, val_dataset, _ = random_split(train_dataset, [train_size, val_size, test_size])

        if args.mode == "train":
            model = training(
                train_dataset, 
                val_dataset,
                args,

                vocab_size = 30522,
                input_dim=64,
                model_dim=32,
                num_heads=8,
                num_classes=1,
                num_layers=1,
                dropout=0.5,
                input_dropout=0.2,
                lr=1e-5,
                warmup=50,
                weight_decay=1e-4
            )
        if args.mode == "test":
            testing(test_dataset)
    except Exception as e:
        logger.exception(e)
    Notice().send("[+] Training finished!")