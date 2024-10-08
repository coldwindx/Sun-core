import collections
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

from tools import Notice
__PATH__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(__PATH__)  

from sampler import ImbalancedDatasetSampler
from dataset import SCDataset, ScDataset, sc_collate_fn

seed = 6
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)                    # 为CPU设置随机种子
torch.cuda.manual_seed(seed)               # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)           # 为所有GPU设置随机种子
pl.seed_everything(seed, workers=True)

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


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)
    
    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]
    
    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class LstmPredictor(pl.LightningModule):
    def __init__(self, vocab_size, input_dim, model_dim, num_classes, num_layers, lr, warmup, max_iters, dropout=0.0, input_dropout=0.0, weight_decay=0.0) -> None:
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
        self.lstm = nn.LSTM(input_size=self.hparams.model_dim, 
                            hidden_size=self.hparams.model_dim, 
                            num_layers=self.hparams.num_layers, batch_first=True, bidirectional=False, 
                            dropout=self.hparams.input_dropout)

        self.output_net = nn.Sequential(
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        """
            x: [batch_size, max_seq_len]输入序列
            lengths: [batch_size]，每个样本的实际长度
        
            返回:
            logits: 分类的logits
        """
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(x, lengths)
        embedded = self.input_net(text)
        sorted_seq_lengths = sorted_seq_lengths.cpu()
        packed_embedded = pack_padded_sequence(embedded, sorted_seq_lengths, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        output = output[desorted_indices]
        batch_size, max_seq_len, hidden_dim = output.shape
        hidden = torch.mean(torch.reshape(hidden,[batch_size,-1,hidden_dim]),dim=1)
        output = torch.sum(output,dim=1)
        logits = self.output_net(output + hidden)
        return logits
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # self.lr_scheduler = CosineWarmupScheduler(optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters)
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
    

class ScPredictor(LstmPredictor):
    def _calculate_loss(self, batch, mode="train"):
        inp_data, _, lengths, labels = batch
        preds = self.forward(inp_data, lengths=lengths)
        preds = preds.reshape(labels.shape)

        loss = F.binary_cross_entropy(preds, labels)
        acc = ((preds >= 0.5) == labels).float().mean()

        self.log("%s_loss" % mode, loss, on_epoch=True)
        self.log("%s_acc" % mode, acc, on_epoch=True)
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
    
CHECKPOINT_PATH = "/home/zhulin/workspace/Sun-core/ckpt"
TEST_DATASETS_PATH = "/home/zhulin/datasets/cdatasets_test.txt"

def training(train_loader, val_loader, **kwargs):
    torch.set_float32_matmul_precision(precision="high")
    root_dir = os.path.join(CHECKPOINT_PATH, "ScLstmTask")
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

    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ScLstmTask.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained mode, loading...")
        model = ScPredictor.load_from_checkpoint(pretrained_filename)
    else:
        model = ScPredictor(max_iters=trainer.max_epochs * len(train_loader), **kwargs)
        trainer.fit(model, train_loader, val_loader)
        # CKPT_PATH = "/home/zhulin/workspace/Sun-core/ckpt/ScPredicTask/lightning_logs/version_3/checkpoints/epoch=37-step=47500.ckpt"
        # trainer.fit(model, train_loader, val_loader, ckpt_path=CKPT_PATH)

    return model

def testing(test_loader, **kwargs):
    torch.set_float32_matmul_precision(precision="high")
    # load model
    CHECKPOINT_PATH = "/home/zhulin/workspace/Sun-core/ckpt/ScLstmTask/lightning_logs/version_1/checkpoints/"

    trainer = pl.Trainer(enable_checkpointing=False, logger=False)
    
    ckpt = f"epoch=15-step=80000.ckpt"
    pretrained_filename = os.path.join(CHECKPOINT_PATH, ckpt)
    classifier = ScPredictor.load_from_checkpoint(pretrained_filename)
    classifier.eval()

    predictions = trainer.predict(classifier, dataloaders=test_loader)
    predictions = torch.cat(predictions, dim=0)
    y_hat = [1 if i >= 0.5 else 0 for i in predictions]
    labels = test_dataset.get_labels()
    
    tp = sum([(x == labels[i] and labels[i] == 1) for i, x in enumerate(y_hat)])
    tn = sum([(x == labels[i] and labels[i] == 0) for i, x in enumerate(y_hat)])
    fp = sum([(x != labels[i] and labels[i] == 0) for i, x in enumerate(y_hat)])
    fn = sum([(x != labels[i] and labels[i] == 1) for i, x in enumerate(y_hat)])
    acc = accurary(tp, tn, fp, fn)
    pre = precision(tp, tn, fp, fn)
    rec = recall(tp, tn, fp, fn)
    fprv = fpr(tp, tn, fp, fn)
    auc = 2 * pre * rec / (pre + rec)
    print(f"tp: {tp}\ntn: {tn}\nfp: {fp}\nfn: {fn}\nacc: {acc}\npre: {pre}\nrec: {rec}\nfpr: {fprv}\nauc: {auc}")

if __name__ == "__main__":

    try:
        train_dataset = SCDataset("/home/zhulin/datasets/cdatasets_train.txt")
        validate_dataset = SCDataset("/home/zhulin/datasets/cdatasets_val.txt")
        # test_dataset = SCDataset("/home/zhulin/datasets/cdatasets_test.txt")
  
        full_dataset = ConcatDataset([train_dataset, validate_dataset])
        train_size = int(0.6 * len(full_dataset))
        val_size = int(0.2 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

        # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=sc_collate_fn, num_workers=4)
        sampler = ImbalancedDatasetSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=sc_collate_fn, num_workers=4, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=sc_collate_fn, num_workers=4)

        model = training(
            train_loader, val_loader,
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