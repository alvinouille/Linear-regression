import numpy as np
MEAN_X = 0
STD_X = 1

def mean(liste) -> float:
    return float(np.sum(liste) / liste.size)

def std(liste, ddof=0) -> float:
    mean_ = mean(liste)
    var = float(np.sum((liste - mean_) ** 2) / (liste.size - ddof))
    return float(np.sqrt(var))

def zscore(x, mean_x, std_x):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    return (x - mean_x) / std_x

def normalisation(x, norm_params):
    mean_x, std_x = norm_params
    return zscore(x, mean_x, std_x)