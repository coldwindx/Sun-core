import argparse
import math
import os
import random
import sys
from loguru import logger
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torcheval.metrics.functional import binary_confusion_matrix, binary_accuracy, binary_precision, binary_recall, binary_f1_score, binary_auroc

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.plugins.environments import SLURMEnvironment
__PATH__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(__PATH__)  

import metrics
from sampler import ImbalancedDatasetSampler
from tools import Notice
from dataset import ScDataset, sc_collate_fn
from network import AttentionPooling, CosineWarmupScheduler, MaskedMeanPooling
from tools import Config

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)                    # 为CPU设置随机种子
torch.cuda.manual_seed(seed)               # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)           # 为所有GPU设置随机种子
pl.seed_everything(seed, workers=True)
torch.set_float32_matmul_precision(precision="high")
CONFIG = Config()

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
        attn_mask = mask.unsqueeze(1).unsqueeze(2)
        attn_logits = attn_logits.masked_fill(attn_mask == 0, -9e15)
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
    def __init__(self, d_model, max_len=4096):
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


class TransformerPredictor(nn.Module):
    def __init__(self, vocab_size, input_dim, model_dim, num_classes, num_heads, num_layers, dropout=0.0, input_dropout=0.0):
        super().__init__()
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Embedding(vocab_size, input_dim),
            nn.Dropout(input_dropout), 
            nn.Linear(input_dim, model_dim)
        )

        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=model_dim)
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            input_dim = model_dim,
            dim_feedforward=2 * model_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        # Output classifier per sequence lement
        self.pooling_net = MaskedMeanPooling()
        # self.pooling_net = AttentionPooling(model_dim, model_dim)
        self.output_net = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            # nn.LayerNorm(model_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
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
        x = self.transformer(x, mask=mask)          # [Batch, SeqLen, ModDim]
        x = self.pooling_net(x, mask=mask)            # GlobalAveragePooling
        x = self.output_net(x)
        return x

class ScPredictor(pl.LightningModule):
    def __init__(self, vocab_size, input_dim, model_dim, num_classes, num_heads, num_layers, lr, warmup, max_iters, dropout=0.0, input_dropout=0.0, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.net = TransformerPredictor(vocab_size, input_dim, model_dim, num_classes, num_heads, num_layers, dropout, input_dropout)

    def forward(self, x, mask=None, add_positional_encoding=True):
         return self.net(x, mask=mask, add_positional_encoding=add_positional_encoding)
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, momentum=0.9)
        scheduler = CosineWarmupScheduler(optimizer, warmup=self.hparams.warmup, epochs=self.hparams.max_iters)
        return [optimizer], [scheduler]
    
    def on_before_optimizer_step(self, optimizer):
        self.log("learning_rate", optimizer.param_groups[0]['lr'], on_step=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        inp_data, mask, _, labels = batch
        preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)
        preds = preds.reshape(labels.shape)

        loss = F.binary_cross_entropy(preds, labels)
        acc = ((preds >= 0.5) == labels).float().mean()

        self.log("train_loss", loss, on_epoch=True, enable_graph=True, sync_dist=True)
        self.log("train_acc", acc, on_epoch=True, enable_graph=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inp_data, mask, _, labels = batch
        preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)
        preds = preds.reshape(labels.shape)

        loss = F.binary_cross_entropy(preds, labels)
        acc = ((preds >= 0.5) == labels).float().mean()

        self.log("valid_loss", loss, on_epoch=True, enable_graph=True, sync_dist=True)
        self.log("valid_acc", acc, on_epoch=True, enable_graph=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        inp_data, mask, _, labels = batch
        preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)
        preds = preds.reshape(labels.shape)

        loss = F.binary_cross_entropy(preds, labels)
        acc = ((preds >= 0.5) == labels).float().mean()

        self.log("test_loss", loss, on_epoch=True, enable_graph=True, sync_dist=True)
        self.log("test_acc", acc, on_epoch=True, enable_graph=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        # padded_sent_seq["input_ids"], padded_sent_seq["attention_mask"], data_length, labels
        inp_data, mask, _, labels = batch
        preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)
        preds = preds.reshape(labels.shape)
        return preds

def training(train_dataset, val_dataset, args, **kwargs):
    root_dir = os.path.join(CONFIG["checkpoint"]["path"], "ScPredicTask")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[
            StochasticWeightAveraging(swa_lrs=1e-2),
            ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
        ],
        # accelerator="gpu", devices=1, num_nodes=1, strategy="ddp",
        accelerator="auto",
        max_epochs=30,
        accumulate_grad_batches=8,
        # limit_train_batches= 4, 
        # limit_val_batches=4,
    )
    trainer.logger._default_hp_metric = None

    sampler = ImbalancedDatasetSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=sc_collate_fn, num_workers=4, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=sc_collate_fn, num_workers=4)

    model = ScPredictor(max_iters=trainer.max_epochs * 4, **kwargs)
    trainer.fit(model, train_loader, val_loader)
    return model

# def testing(test_dataset, args, **kwargs):
#     if args.ckpt:
#         pretrained_filename = CONFIG["checkpoint"]["path"] + "ScPredicTask/lightning_logs"
#         pretrained_filename = pretrained_filename + f"/version_{args.version}/checkpoints/{args.ckpt}"
#     else:
#         pretrained_filename = CONFIG["checkpoint"]["path"] + "ScPredicTask/lightning_logs" + CONFIG["checkpoint"]["ScPredicTask"]
#     trainer = pl.Trainer(enable_checkpointing=False, logger=False)
#     classifier = ScPredictor.load_from_checkpoint(pretrained_filename)
#     classifier.eval()

#     test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=sc_collate_fn, num_workers=4)
#     predictions = trainer.predict(classifier, dataloaders=test_loader)
#     predictions = torch.cat(predictions, dim=0)
#     y_hat = [1 if i >= 0.5 else 0 for i in predictions]
    
#     tn, fp, fn, tp = confusion_matrix(test_dataset.labels, y_hat).ravel()
#     acc = metrics.accurary(tp, tn, fp, fn)
#     pre = metrics.precision(tp, tn, fp, fn)
#     rec = metrics.recall(tp, tn, fp, fn)
#     fprv = metrics.fpr(tp, tn, fp, fn)
#     auc = 2 * pre * rec / (pre + rec)
#     print(f"tp: {tp}\ntn: {tn}\nfp: {fp}\nfn: {fn}\nacc: {acc}\npre: {pre}\nrec: {rec}\nfpr: {fprv}\nauc: {auc}")

def testing(test_dataset, args, **kwargs):
    pretrained_filename = CONFIG["checkpoint"]["path"] + "ScPredicTask/lightning_logs" + \
                            f"/version_{args.version}/checkpoints/{args.ckpt}" if args.ckpt else CONFIG["checkpoint"]["ScPredicTask"]

    ### load model
    trainer = pl.Trainer(enable_checkpointing=False, logger=False)
    classifier = ScPredictor.load_from_checkpoint(pretrained_filename)
    classifier.eval()
    
    ### load dataset
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=sc_collate_fn, num_workers=4)

    ### predict scores
    scores = trainer.predict(classifier, dataloaders=test_loader)
    scores = torch.cat(scores, dim=0)

    ### binary_confusion_matrix
    labels = torch.tensor([test_dataset.dataset[i][1] for i in test_dataset.indices], device="cuda")
    cm = binary_confusion_matrix(scores, labels)
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    accuracy = binary_accuracy(scores, labels, threshold = 0.5)
    precision = binary_precision(scores, labels, threshold = 0.5).item()
    recall = binary_recall(scores, labels, threshold = 0.5).item()
    f1 = binary_f1_score(scores, labels, threshold = 0.5).item()
    auc = binary_auroc(scores, labels).item()

    print(f"tp: {tp}\ntn: {tn}\nfp: {fp}\nfn: {fn}\nacc: {accuracy}\npre: {precision}\nrec: {recall}\nauc: {auc}\nf1: {f1}")
if __name__ == "__main__":

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', default='train', type=str)
        parser.add_argument('--version', default='0', type=str)
        parser.add_argument('--ckpt', default='epoch=29-step=2999010.ckpt', type=str)
        args = parser.parse_args()

        full_dataset = ScDataset(CONFIG["datasets"]["train"])
        test_dataset = ScDataset(CONFIG["datasets"]["test"])

        val_size = 1024 * 64
        test_size = 1024 * 64
        train_size = len(full_dataset) - val_size - test_size
        train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, test_size])

        if args.mode == "train":
            model = training(
                train_dataset, 
                val_dataset,
                args,
                vocab_size = 30522,
                input_dim=64,
                model_dim=64,
                num_heads=8,
                num_classes=1,
                num_layers=1,
                dropout=0.1,
                input_dropout=0.1,
                lr=1e-4,
                warmup=2 * 4,
                weight_decay=1e-6
            )
        if args.mode == "test":
            testing(test_dataset, args)
    except Exception as e:
        logger.exception(e)
    Notice().send("[+] transformer.py execute finished!")