import collections
import os
import random
import sys
from loguru import logger
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

def main(ckpt):
    torch.set_float32_matmul_precision(precision="high")
    # load model
    CHECKPOINT_PATH = "/home/zhulin/workspace/Sun-core/ckpt/ScPredicTask/lightning_logs/version_0/checkpoints/"
    TEST_DATASETS_PATH = "/home/zhulin/datasets/cdatasets_test.txt"

    test_dataset = SCDataset(TEST_DATASETS_PATH)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=sc_collate_fn, num_workers=4)

    for batch in test_loader:
        pass
if __name__ == "__main__":
    try:
        # main()
        main(ckpt="")
    except Exception as e:
        logger.exception(e)
    # Notice().send("[+] Test finished!")