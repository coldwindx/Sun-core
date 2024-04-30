import math
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
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
__PATH__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(__PATH__)  

from sampler import ImbalancedDatasetSampler
from tools import Notice
from dataset import ScDataset, sc_collate_fn
from network import MaskedMeanPooling
from tools import Config

seed = 6
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)                    # 为CPU设置随机种子
torch.cuda.manual_seed(seed)               # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)           # 为所有GPU设置随机种子
pl.seed_everything(seed, workers=True)
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
        self.lr_scheduler.step()
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    def test_step(self, batch, batch_idx):
        raise NotImplementedError


class DeepRanPredictor(DeepRan):
    def _calculate_loss(self, batch, mode="train"):
        inp_data, mask, _, labels = batch
        preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)
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
        inp_data, mask, _, labels = batch
        preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)
        preds = preds.reshape(labels.shape)
        return preds

def training(train_loader, val_loader, checkpoint, **kwargs):
    torch.set_float32_matmul_precision(precision="high")
    root_dir = os.path.join(checkpoint, "ScDeepRanTask")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[
            # StochasticWeightAveraging(swa_lrs=1e-2),
            ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
        ],
        accelerator="auto",
        # precision="bf16-true",
        devices=1,
        max_epochs=80,
        # accumulate_grad_batches=4,
        # gradient_clip_val=10,
        limit_train_batches= 5000, 
        # limit_val_batches=5000,
        # enable_progress_bar=False
    )
    trainer.logger._default_hp_metric = None

    model = DeepRanPredictor(max_iters=trainer.max_epochs * len(train_loader), **kwargs)
    trainer.fit(model, train_loader, val_loader)

    return model

if __name__ == "__main__":

    try:
        train_dataset = ScDataset(CONFIG["datasets"]["train"])
        validate_dataset = ScDataset(CONFIG["datasets"]["validate"])
        test_dataset = ScDataset(CONFIG["datasets"]["test"])

        full_dataset = ConcatDataset([train_dataset, validate_dataset])
        train_size = int(0.6 * len(full_dataset))
        val_size = int(0.2 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

        sampler = ImbalancedDatasetSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=sc_collate_fn, num_workers=4, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=sc_collate_fn, num_workers=4)

        model = training(
            train_loader, 
            val_loader,
            CONFIG["checkpoint"]["ScDeepRanTask"],
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

    except Exception as e:
        logger.exception(e)
    Notice().send("[+] Training finished!")