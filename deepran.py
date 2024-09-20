import argparse
import os
import random
import sys
from loguru import logger
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import confusion_matrix
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
__PATH__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(__PATH__)  

import metrics
from sampler import ImbalancedDatasetSampler
from dataset import ScDataset, sc_collate_fn
from tools import Config, Notice

seed = 6
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)                    # 为CPU设置随机种子
torch.cuda.manual_seed(seed)               # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)           # 为所有GPU设置随机种子
pl.seed_everything(seed, workers=True)
torch.set_float32_matmul_precision(precision="high")
CONFIG = Config()

def prepare_pack_padded_sequence(inputs_words, seq_lengths, descending=True):
    """
    for rnn model
    :param device:
    :param inputs_words:
    :param seq_lengths:
    :param descending:
    :return:
    """
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=descending)
    _, desorted_indices = torch.sort(indices, descending=False)
    sorted_inputs_words = inputs_words[indices]
    return sorted_inputs_words, sorted_seq_lengths, desorted_indices

class BiLstm(pl.LightningModule):
    def __init__(self, vocab_size, input_dim, model_dim, num_classes, num_layers, lr, warmup, max_iters, dropout=0.0, input_dropout=0.0, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Embedding(self.hparams.vocab_size, self.hparams.input_dim),
            nn.Dropout(self.hparams.input_dropout), 
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        )

        # BiLstm
        self.lstm = nn.LSTM(
            input_size=self.hparams.model_dim,
            hidden_size=self.hparams.model_dim,
            num_layers=self.hparams.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.hparams.dropout
        )
        # Output classifier per sequence lement
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.w = nn.Parameter(torch.randn(self.hparams.model_dim * 2), requires_grad=True)
        self.fc = nn.Sequential(
            nn.Dropout(self.hparams.dropout, inplace=True),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes)
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(x, lengths)
        embedded = self.input_net(text)
        sorted_seq_lengths = sorted_seq_lengths.cpu()
        packed_embedded = pack_padded_sequence(embedded, sorted_seq_lengths, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        output = output[desorted_indices]

        alpha = F.softmax(torch.matmul(self.tanh1(output), self.w), dim=0).unsqueeze(-1)
        output_atten = output * alpha
        output_atten = torch.sum(output_atten,dim=1)

        batch_size, max_seq_len, hidden_dim = output.shape
        hidden = torch.mean(torch.reshape(hidden,[batch_size,-1,hidden_dim]),dim=1)

        output = torch.sum(output, dim=1)
        logits = self.fc(output + output_atten + hidden)
        return logits

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        # self.lr_scheduler.step()
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    def test_step(self, batch, batch_idx):
        raise NotImplementedError

class DeepRan(pl.LightningModule):
    def __init__(self, vocab_size, input_dim, model_dim, num_classes, num_layers, lr, warmup, max_iters, dropout=0.0, input_dropout=0.0, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Embedding(self.hparams.vocab_size, self.hparams.input_dim),
            nn.Dropout(self.hparams.input_dropout), 
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        )

        # BiLstm
        self.lstm = nn.LSTM(
            input_size=self.hparams.model_dim,
            hidden_size=self.hparams.model_dim,
            num_layers=self.hparams.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.hparams.dropout
        )
        # Output classifier per sequence lement
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.randn(self.hparams.num_layers * 2, self.hparams.model_dim), requires_grad=True)
        self.fc = nn.Sequential(
            nn.Dropout(self.hparams.dropout, inplace=True),
            nn.Linear(self.hparams.num_layers * 2, self.hparams.num_classes),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        import pdb
        pdb.set_trace()
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(x, lengths)
        embedded = self.input_net(text)
        sorted_seq_lengths = sorted_seq_lengths.cpu()
        packed_embedded = pack_padded_sequence(embedded, sorted_seq_lengths, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        alpha = torch.sum(torch.mul(self.w, hidden.permute(1, 0, 2)), dim=2)
        logits = self.fc(alpha)
        return logits

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        # self.lr_scheduler.step()
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    def test_step(self, batch, batch_idx):
        raise NotImplementedError


class DeepRanPredictor(DeepRan):
    def _calculate_loss(self, batch, mode="train"):
        inp_data, _, lengths, labels = batch
        preds = self.forward(inp_data, lengths=lengths)
        preds = preds.reshape(labels.shape)

        loss = F.binary_cross_entropy(preds, labels)
        acc = ((preds >= 0.5) == labels).float().mean()

        self.log("%s_loss" % mode, loss, on_epoch=True, enable_graph=True)
        self.log("%s_acc" % mode, acc, on_epoch=True, enable_graph=True)
        return loss, acc

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
        # padded_sent_seq["input_ids"], padded_sent_seq["attention_mask"], data_length, labels
        inp_data, _, lengths, labels = batch
        preds = self.forward(inp_data, lengths=lengths)
        preds = preds.reshape(labels.shape)
        return preds

def training(train_dataset, val_dataset, args, **kwargs):
    root_dir = os.path.join(CONFIG["checkpoint"]["path"], "ScDeepRanTask")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[
            ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
        ],
        accelerator="auto",
        devices=1,
        max_epochs=30,
        limit_train_batches= 5000
    )
    trainer.logger._default_hp_metric = None

    sampler = ImbalancedDatasetSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=sc_collate_fn, num_workers=4, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=sc_collate_fn, num_workers=4)

    model = DeepRanPredictor(max_iters=trainer.max_epochs * len(train_loader), **kwargs)
    if args.enable_ckpt:
        pretrained_filename = CONFIG["checkpoint"]["path"] + "ScDeepRanTask/lightning_logs" + CONFIG["checkpoint"]["ScDeepRanTask"]
        trainer.fit(model, train_loader, val_loader, ckpt_path=pretrained_filename)
    else:
        trainer.fit(model, train_loader, val_loader)

    return model

def testing(test_dataset, **kwargs):
    # load model
    pretrained_filename = CONFIG["checkpoint"]["path"] + "ScDeepRanTask/lightning_logs" + CONFIG["checkpoint"]["ScDeepRanTask"]

    trainer = pl.Trainer(enable_checkpointing=False, logger=False)
    classifier = DeepRanPredictor.load_from_checkpoint(pretrained_filename)
    classifier.eval()

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=sc_collate_fn, num_workers=4)
    predictions = trainer.predict(classifier, dataloaders=test_loader)
    predictions = torch.cat(predictions, dim=0)
    y_hat = [1 if i >= 0.5 else 0 for i in predictions]

    tn, fp, fn, tp = confusion_matrix(test_dataset.labels, y_hat).ravel()
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

        full_dataset = ScDataset(CONFIG["datasets"]["train"])
        test_dataset = ScDataset(CONFIG["datasets"]["test"])

        train_size = int(0.6 * len(full_dataset))
        val_size = int(0.2 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, test_size])

        if args.mode == "train":
            model = training(
                train_dataset, 
                val_dataset,
                args,
                vocab_size = 30522,
                input_dim=256,
                model_dim=128,
                num_classes=1,
                num_layers=8,
                dropout=0.5,
                input_dropout=0.2,
                lr=1e-6,
                warmup=50,
                weight_decay=0.0
            )
        if args.mode == "test":
            testing(test_dataset)

    except Exception as e:
        logger.exception(e)
    Notice().send("[+] deepran.py execute finished!")