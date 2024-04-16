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
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
__PATH__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(__PATH__)  
from sampler import ImbalancedDatasetSampler
from tools import Notice
from dataset import SCDataset, sc_collate_fn

seed = 6
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)                    # 为CPU设置随机种子
torch.cuda.manual_seed(seed)               # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)           # 为所有GPU设置随机种子
SEED = torch.Generator().manual_seed(seed)
pl.seed_everything(seed, workers=True)

def scaled_dot_product(q, k, v, mask=None):
    '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
    '''
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 module number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
    
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention: 
            return o, attention
        return o
    
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        # x: [BatchSize, SeqLen, InputDim] -> [SeqLen, BatchSize, InputDim]
        x = x.permute(1, 0, 2)
        scores = self.U(torch.tanh(self.W(x)))
        weights = F.softmax(scores, dim=0)
        pooled = torch.sum(x * weights, dim=0)
        return pooled
    
class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """EncoderBlock.

        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.SiLU(inplace=True),      # 不会梯度消失
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)
    
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x

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
    
class TransformerPredictor(pl.LightningModule):
    def __init__(self, vocab_size, input_dim, model_dim, num_classes, num_heads, num_layers, lr, warmup, max_iters, dropout=0.0, input_dropout=0.0):
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
        # self.input_net = nn.Embedding(self.hparams.vocab_size, self.hparams.input_dim)
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            input_dim = self.hparams.model_dim,
            dim_feedforward=2 * self.hparams.model_dim,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout
        )
        # Output classifier per sequence lement
        self.output_net = nn.Sequential(
            # nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            AttentionPooling(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """

        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
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


class ScPredictor(TransformerPredictor):
    def _calculate_loss(self, batch, mode="train"):
        inp_data, mask, _, labels = batch
        mask = mask.unsqueeze(1).unsqueeze(2)
        preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)
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
        inp_data, mask, _, labels = batch
        mask = mask.unsqueeze(1).unsqueeze(2)
        preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)
        preds = preds.reshape(labels.shape)
        return preds
    
CHECKPOINT_PATH = "/home/zhulin/workspace/Sun/ckpt/"

def training(train_loader, val_loader, **kwargs):
    torch.set_float32_matmul_precision(precision="high")
    root_dir = os.path.join(CHECKPOINT_PATH, "ScPredicTask")
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
        max_epochs=120,
        accumulate_grad_batches=8,
        # gradient_clip_val=10,
        limit_train_batches= 5000, 
        # limit_val_batches=5000,
        
    )
    trainer.logger._default_hp_metric = None

    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ScPredicTask.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained mode, loading...")
        model = ScPredictor.load_from_checkpoint(pretrained_filename)
    else:
        model = ScPredictor(max_iters=trainer.max_epochs * len(train_loader), **kwargs)
        # ckpt = "/home/zhulin/workspace/Sun/ckpt/ScPredicTask/lightning_logs/version_7/checkpoints/epoch=99-step=62500.ckpt"
        trainer.fit(model, train_loader, val_loader)
    
    return model

if __name__ == "__main__":

    try:
        train_dataset = SCDataset("/mnt/sdd1/data/sun/cdatasets_train.txt")
        validate_dataset = SCDataset("/mnt/sdd1/data/sun/cdatasets_val.txt")

        train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=sc_collate_fn, num_workers=4, 
                                  sampler=ImbalancedDatasetSampler(train_dataset))
        val_loader = DataLoader(validate_dataset, batch_size=64, shuffle=False, collate_fn=sc_collate_fn, num_workers=4)

        model = training(
            train_loader, val_loader,
            vocab_size = 30522,
            input_dim=64,
            model_dim=32,
            num_heads=8,
            num_classes=1,
            num_layers=1,
            dropout=0.4,
            lr=1e-5,
            warmup=50,
        )

    except Exception as e:
        logger.exception(e)
    Notice().send("[+] Training finished!")