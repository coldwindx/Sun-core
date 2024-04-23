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
from torch.nn.utils.rnn import pack_padded_sequence
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
__PATH__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(__PATH__)  

from sampler import ImbalancedDatasetSampler
from dataset import SCDataset, sc_collate_fn

seed = 6
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)                    # 为CPU设置随机种子
torch.cuda.manual_seed(seed)               # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)           # 为所有GPU设置随机种子
pl.seed_everything(seed, workers=True)

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
            nn.Linear(self.hparams.model_dim * self.hparams.num_layers, self.hparams.num_classes),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        """
            x: [batch_size, max_seq_len, input_size]，已经padding的输入序列
            lengths: [batch_size]，每个样本的实际长度
        
            返回:
            logits: 分类的logits
        """
        x = self.input_net(x)   #[batch_size, max_seq_len, input_size]，已经padding的输入序列
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # 通过LSTM hidden: num_layers * num_directions, batch, hidden_size
        packed_output, (hidden, cell) = self.lstm(x)
        # 取最后一个有效时间步的隐藏状态作为整个序列的表示
        # 注意：对于双向LSTM，hidden将是(num_layers * num_directions, batch, hidden_size)，需要进一步处理
        hidden = torch.transpose(hidden, 0, 1).contiguous()
        hidden = hidden.view(hidden.shape[0], -1)
        # 通过全连接层得到分类结果
        logits = self.output_net(hidden)
        
        return logits
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.lr_scheduler = CosineWarmupScheduler(optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters)
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

def training(train_loader, val_loader, **kwargs):
    torch.set_float32_matmul_precision(precision="high")
    root_dir = os.path.join(CHECKPOINT_PATH, "ScLstmTask")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[
            StochasticWeightAveraging(swa_lrs=1e-2),
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
            input_dim=64,
            model_dim=32,
            num_classes=1,
            num_layers=32,
            dropout=0.5,
            input_dropout=0.2,
            lr=1e-5,
            warmup=50,
            weight_decay=1e-4
        )

    except Exception as e:
        logger.exception(e)
    # Notice().send("[+] Training finished!")