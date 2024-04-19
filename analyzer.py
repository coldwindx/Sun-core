import os
import random
import sys
from loguru import logger
import lightning as pl
from matplotlib import pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

__PATH__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(__PATH__)  

seed = 6
random.seed(seed)
torch.manual_seed(seed)                    # 为CPU设置随机种子
torch.cuda.manual_seed(seed)               # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)           # 为所有GPU设置随机种子
SEED = torch.Generator().manual_seed(seed)
pl.seed_everything(seed, workers=True)

TEST_DATASETS_PATH = "/home/zhulin/datasets/cdatasets_test.txt.d"
PRETRAIN_PATH = "/home/zhulin/pretrain/"

def distribution():
    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_PATH + 'bert_pretrain_uncased/')
    # 读取文本文件并分句，并统计句子长度
    length_distribution = {}
    with open(TEST_DATASETS_PATH, 'r') as f:
        cnt: int = 0
        for line in f:
            cnt += 1
            # if cnt > 2000:
            #     break
            if cnt % 2 == 1:
                # 第一行: [索引 UniqueKey PID 标签]
                continue
            if cnt % 2 == 0:
                padded_sent_seq = tokenizer(line, return_tensors="pt")
                length = len(padded_sent_seq["input_ids"][0])
                length_distribution[length] = length_distribution.get(length, 0) + 1
                continue

    # 转换为可用于绘图的数据格式
    sorted_lengths, counts = zip(*sorted(length_distribution.items()))

    # 计算分位数
    percentiles = [np.percentile(sorted_lengths, p) for p in [90, 95, 99]]

    # 绘制折线图
    plt.plot(sorted_lengths, counts)
    plt.xlabel('Sentence Length')
    plt.ylabel('Number of Sentences')
    plt.title('Distribution of Sentence Lengths')

    # 输出分位数
    print(f"90th Percentile: {percentiles[0]}")
    print(f"95th Percentile: {percentiles[1]}")
    print(f"99th Percentile: {percentiles[2]}")

    # 显示图形
    plt.savefig("distribution.png")
    plt.show()

def fp_analyzer(sample):
    datas, labels, pids, indexes = [], [], [], []
    with open(TEST_DATASETS_PATH, 'r') as f:
        txt = f.readlines()
        cnt: int = 0
        lenght: int = len(txt)
        while cnt < lenght:
            if cnt % 2 == 1:
                # 第一行: [索引 UniqueKey PID 标签]
                index, unique_key, pid, label = txt[cnt].split("\t")
                indexes.append(index)
                pids.append(int(pid))
                labels.append(int(label))
                cnt += 1
                # 数据
                cnt += 1
                datas.append(txt[cnt])
    
    idxes = []
    for i, index in enumerate(indexes):
        if index == sample:
            idxes.append(i)

    for i in idxes:
        if pids[i] != 5052:
            continue
        print(datas[i])

if __name__ == "__main__":
    try:
        # distribution()
        fp_analyzer("kad29f77ee86ed9827158347befa8998d")
    except Exception as e:
        logger.exception(e)
    # Notice().send("[+] Test finished!")