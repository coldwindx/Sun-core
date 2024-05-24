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
