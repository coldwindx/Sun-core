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


    samples = collections.defaultdict(list)
    pkeys = []
    with open(TEST_DATASETS_PATH, 'r') as f:
        cnt: int = 0
        for line in f:
            cnt += 1
            if cnt % 2 == 1:
                # 第一行: [索引 [UniqueKey] 标签]
                index, ukey, pid, label = line.split("\t")
                samples[index].append(cnt // 2)
                pkeys.append(index + "_" + str(ukey) + "_" + str(pid))
                # print(f"{index}, {str(ukey) + '_' + str(pid)}, {label}, {y_hat[cnt // 2]}, {predictions[cnt // 2]}")
    test_samples = ["kad29f77ee86ed9827158347befa8998d", "k2218db42c1b69db72d7432c8d6fcab9d", "kcc378f899d56f8d3c76b9905b47a84a6", "k74d9610a72fa9ed105c927e3b1897c5b", "kba67dd5ab7d6061704f2903573cec303", "k5e271dbfb5803f600b30f7d9945024fd", "kc64eb31c168a78c8b17198b15ba7e638", "k38393408898e353857a18f481cf15935", "kc9ec0d9ff44f445ce5614cc87398b38d", "k21a563f958b73d453ad91e251b11855c", "k643c8c25fbe8c3cc7576bc8e7bcd8a68", "k81fc90c9f339042edc419e0a62a03e17", "k80d2cfccef17caa46226147c1b0648e6", "kdeebbea18401e8b5e83c410c6d3a8b4e", "k732a229132d455b98038e5a23432385d", "kdffd2b26085eddb88743ae3fc7be9eee", "k6992dd450b7581d7c2a040d15610a8c5", "k0c4502d6655264a9aa420274a0ddeaeb", "k209a288c68207d57e0ce6e60ebf60729", "k6e080aa085293bb9fbdcc9015337d309", "k58b70be83f9735f4e626054de966cc94", "keba85b706259f4dc0aec06a6a024609a", "kc24f6144e905b717a372c529d969611e", "k0a47084d98bed02037035d8e3120c241", "k087f42dd5c17b7c42723dfc150a8da42", "ke3dd1eb73e602ea95ad3e325d846d37c", "k33a7c3fe6c663032798a6780bb21599c", "k4edfdc708fb7cb3606ca68b6c288f979", "k77d0a95415ef989128805252cba93dc2", "k168447d837fc71deeee9f6c15e22d4f4", "k6c660f960daac148be75427c712d0134", "k84c82835a5d21bbcf75a61706d8ab549", "kb65b194c6cc134d56ba3acdcc7bd3051", "kd5fee0c6f1d0d730de259c64e6373a0c", "k1de48555aafd904f53e8b19f99658ce8", "k64497a0fa912f0e190359684de92be2d", "k2bbb2d9be1a993a8dfef0dd719c589a0", "ke4e439fc5ade188ba2c69367ba6731b6", "kc24f6144e905b717a372c529d969611e", "ke1e41506da591e55cee1825494ac8f42", "k2bbff2111232d73a93cd435300d0a07e", "k8c64c2ff302f64cf326897af8176d68e", "k00e3b3952d6cfe18aba4554a034f8e55", "kb7be2da288647b28c1697615e8d07b17", "kb572a0486274ee9c0ba816c1b91b87c7", "k25a54e24e9126fba91ccb92143136e9f", "ke3f6878bcafe2463f6028956f44a6e74", "k0880430c257ce49d7490099d2a8dd01a", "k5c7fb0927db37372da25f270708103a2", "k9ce01dfbf25dfea778e57d8274675d6f"]
    # test_samples = ["kad29f77ee86ed9827158347befa8998d"]

    stp, stn, sfp, sfn = 0, 0, 0, 0
    for index in test_samples:
        plabel = collections.defaultdict(int)
        psign = collections.defaultdict(int)
        # 规约
        for id in samples[index]:
            pkey, label, sign = pkeys[id], labels[id], y_hat[id]
            plabel[pkey] = label
            psign[pkey] = psign[pkey] or sign
        # 指标汇总
        truths = [label for _, label in plabel.items()]
        signs = [psign[pkey] for pkey, _ in plabel.items()]
        tn, fp, fn, tp = confusion_matrix(truths, signs).ravel()
        stp += tp
        stn += tn
        sfp += fp
        sfn += fn
        acc = accurary(tp, tn, fp, fn)
        pre = precision(tp, tn, fp, fn)
        rec = recall(tp, tn, fp, fn)
        fprv = fpr(tp, tn, fp, fn)
        auc = 2 * pre * rec / (pre + rec)
        print(f"{index}\t{tp}\t{tn}\t{fp}\t{fn}\t{acc}\t{pre}\t{rec}\t{fprv}\t{auc}")
    acc = accurary(stp, stn, sfp, sfn)
    pre = precision(stp, stn, sfp, sfn)
    rec = recall(stp, stn, sfp, sfn)
    fprv = fpr(stp, stn, sfp, sfn)
    auc = 2 * pre * rec / (pre + rec)
    print(f"ALL\t{stp}\t{stn}\t{sfp}\t{sfn}\t{acc}\t{pre}\t{rec}\t{fprv}\t{auc}")
   

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