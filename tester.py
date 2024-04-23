import collections
import os
import random
import sys
from loguru import logger
from sklearn.metrics import accuracy_score, confusion_matrix
import lightning as pl
import torch
from torch.utils.data import DataLoader
__PATH__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(__PATH__)  
from transformer import ScPredictor
from dataset import SCDataset, sc_collate_fn

seed = 6
random.seed(seed)
torch.manual_seed(seed)                    # 为CPU设置随机种子
torch.cuda.manual_seed(seed)               # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)           # 为所有GPU设置随机种子
SEED = torch.Generator().manual_seed(seed)
pl.seed_everything(seed, workers=True)

def accurary(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)
def precision(tp, tn, fp, fn):
    return tp / (tp + fp)
def recall(tp, tn, fp, fn):
    return tp / (tp + fn)
def fpr(tp, tn, fp, fn):
    if fp + tn == 0:
        return 0.0
    return fp / (fp + tn)

def test(ckpt):
    torch.set_float32_matmul_precision(precision="high")
    # load model
    CHECKPOINT_PATH = "/home/zhulin/workspace/Sun-core/ckpt/ScPredicTask/lightning_logs/version_1/checkpoints/"

    trainer = pl.Trainer(enable_checkpointing=False, logger=False)
    
    TEST_DATASETS_PATH = "/home/zhulin/datasets/cdatasets_test.txt"
    test_dataset = SCDataset(TEST_DATASETS_PATH)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=sc_collate_fn, num_workers=4)

    ckpt = f"epoch=74-step=46875.ckpt"
    pretrained_filename = os.path.join(CHECKPOINT_PATH, ckpt)
    classifier = ScPredictor.load_from_checkpoint(pretrained_filename)
    classifier.eval()

    predictions = trainer.predict(classifier, dataloaders=test_loader)
    predictions = torch.cat(predictions, dim=0)
    y_hat = [1 if i >= 0.5 else 0 for i in predictions]
    labels = test_dataset.get_labels()
    
    samples = []
    pkeys = []
    with open(TEST_DATASETS_PATH, 'r') as f:
        cnt: int = 0
        for line in f:
            cnt += 1
            if cnt % 2 == 1:
                # 第一行: [索引 [UniqueKey] 标签]
                index, ukey, pid, label = line.split("\t")
                samples.append(index)
                pkeys.append(ukey + "_" + pid)
    test_samples = ["kad29f77ee86ed9827158347befa8998d", "k2218db42c1b69db72d7432c8d6fcab9d", "kcc378f899d56f8d3c76b9905b47a84a6", "k74d9610a72fa9ed105c927e3b1897c5b", "kba67dd5ab7d6061704f2903573cec303", "k5e271dbfb5803f600b30f7d9945024fd", "kc64eb31c168a78c8b17198b15ba7e638", "k38393408898e353857a18f481cf15935", "kc9ec0d9ff44f445ce5614cc87398b38d", "k21a563f958b73d453ad91e251b11855c", "k643c8c25fbe8c3cc7576bc8e7bcd8a68", "k81fc90c9f339042edc419e0a62a03e17", "k80d2cfccef17caa46226147c1b0648e6", "kdeebbea18401e8b5e83c410c6d3a8b4e", "k732a229132d455b98038e5a23432385d", "kdffd2b26085eddb88743ae3fc7be9eee", "k6992dd450b7581d7c2a040d15610a8c5", "k0c4502d6655264a9aa420274a0ddeaeb", "k209a288c68207d57e0ce6e60ebf60729", "k6e080aa085293bb9fbdcc9015337d309", "k58b70be83f9735f4e626054de966cc94", "keba85b706259f4dc0aec06a6a024609a", "kc24f6144e905b717a372c529d969611e", "k0a47084d98bed02037035d8e3120c241", "k087f42dd5c17b7c42723dfc150a8da42", "ke3dd1eb73e602ea95ad3e325d846d37c", "k33a7c3fe6c663032798a6780bb21599c", "k4edfdc708fb7cb3606ca68b6c288f979", "k77d0a95415ef989128805252cba93dc2", "k168447d837fc71deeee9f6c15e22d4f4", "k6c660f960daac148be75427c712d0134", "k84c82835a5d21bbcf75a61706d8ab549", "kb65b194c6cc134d56ba3acdcc7bd3051", "kd5fee0c6f1d0d730de259c64e6373a0c", "k1de48555aafd904f53e8b19f99658ce8", "k64497a0fa912f0e190359684de92be2d", "k2bbb2d9be1a993a8dfef0dd719c589a0", "ke4e439fc5ade188ba2c69367ba6731b6", "kc24f6144e905b717a372c529d969611e", "ke1e41506da591e55cee1825494ac8f42", "k2bbff2111232d73a93cd435300d0a07e", "k8c64c2ff302f64cf326897af8176d68e", "k00e3b3952d6cfe18aba4554a034f8e55", "kb7be2da288647b28c1697615e8d07b17", "kb572a0486274ee9c0ba816c1b91b87c7", "k25a54e24e9126fba91ccb92143136e9f", "ke3f6878bcafe2463f6028956f44a6e74", "k0880430c257ce49d7490099d2a8dd01a", "k5c7fb0927db37372da25f270708103a2", "k9ce01dfbf25dfea778e57d8274675d6f"]

    tp, tn, fp, fn = 0, 0, 0, 0
    for sample in test_samples:
        idx = []
        for i, s in enumerate(samples):
            if s == sample:
                idx.append(i)
        plabel = collections.defaultdict(int)
        psign = collections.defaultdict(int)
        for id in idx:
            pkey, label, sign = pkeys[id], labels[id], y_hat[id]
            plabel[pkey] = label
            psign[pkey] = psign[pkey] or sign
        for pkey, label in plabel.items():
            sign = psign[pkey]
            if 0 == label:
                if 0 == sign:
                    tn += 1
                else:
                    fp += 1
            else:
                if 0 == sign:
                    fn += 1
                else:
                    tp += 1
        acc = accurary(tp, tn, fp, fn)
        pre = precision(tp, tn, fp, fn)
        rec = recall(tp, tn, fp, fn)
        fprv = fpr(tp, tn, fp, fn)
        auc = 2 * pre * rec / (pre + rec)
        print(f"{sample}\t{tp}\t{tn}\t{fp}\t{fn}\t{acc}\t{pre}\t{rec}\t{fprv}\t{auc}")

def main(ckpt):
    torch.set_float32_matmul_precision(precision="high")
    # load model
    CHECKPOINT_PATH = "/home/zhulin/workspace/Sun-core/ckpt/ScPredicTask/lightning_logs/version_4/checkpoints/"

    trainer = pl.Trainer(enable_checkpointing=False, logger=False)
    
    TEST_DATASETS_PATH = "/home/zhulin/datasets/cdatasets_test.txt"
    test_dataset = SCDataset(TEST_DATASETS_PATH)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=sc_collate_fn, num_workers=4)

    ckpt = f"epoch=70-step=88750.ckpt"
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
        main(ckpt="")
    except Exception as e:
        logger.exception(e)
    # Notice().send("[+] Test finished!")