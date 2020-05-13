import numpy as np

def normalize_p(p):
    p = np.clip(p, 0, 1)
    p = p / np.sum(p)
    return p
