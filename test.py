
import numpy as np

path = "/mnt/sdd1/data/zhulin/jack/datasets/X.txt"

f = open(path, "r")
X = []
for line in f.readlines():
    X.append(line.split(","))
X = np.array(X, dtype=np.int32)
f.close()

f = open("/mnt/sdd1/data/zhulin/jack/datasets/L.txt", "r")
L = []
for line in f.readlines():
    L.append(line.split(","))
L = np.array(L, dtype=np.int32)
f.close()

np.savez("/mnt/sdd1/data/zhulin/jack/datasets/deepguard.train", X = X, label = L)
import pdb
pdb.set_trace()