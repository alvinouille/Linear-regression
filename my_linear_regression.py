import numpy as np
from tqdm import tqdm

class MyLinearRegression():
    """ Description: My personnal linear regression class to fit like a boss. """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        thetas = thetas.reshape(-1, 1).astype(np.float64)
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def fit_(self, x, y):
        with tqdm(total=self.max_iter, dynamic_ncols=True, desc=f'Training model', bar_format="ETA: {remaining}s [{l_bar}][{bar}] {n_fmt}/{total_fmt} | time {elapsed}s") as pbar:
            for _ in range(self.max_iter):
                self.thetas -= self.alpha * self.gradient_(x, y, self.thetas)
                pbar.update(1)
        self.thetas = self.thetas.astype(np.float64) 
        return self.thetas.astype(np.float64) 

    
    def predict_(self, x):
        Xprime = np.c_[np.ones(x.shape[0]), x]
        return Xprime @ self.thetas

    def gradient_(self, x, y, theta):
        Xprime = np.c_[np.ones(x.shape[0]), x]
        y_hat = Xprime @ theta
        return Xprime.T @ (y_hat - y) / x.shape[0]

    def loss_(self, y, y_hat):
        m = y.shape[0]
        return sum((y_hat - y) ** 2) / (2 * m)

    def loss_elem_(self, y, y_hat):
        J_elem = y_hat - y
        return J_elem ** 2

    def rmse_(self, y, y_hat):
        J_value = sum(self.loss_elem_(y, y_hat)) / y.shape[0]
        return np.sqrt(float(J_value))

