import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR
import sys
from utils import normalisation, mean, std

def plot_lr(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray,  xlabel: str,
        colors: tuple, lr_model: MyLR):
    plt.xlabel(f"x: {xlabel}")
    plt.ylabel("y: sell price (in keuros)")
    plt.grid()

    plt.plot(x, y, "o", color=colors[0], label="Sell price")
    plt.plot(x, y_hat, "o", color=colors[1], label="Predicted sell price")

    rmse = f"{lr_model.rmse_(y, y_hat):.3f}"
    plt.title(f"RMSE: {rmse}")
    plt.legend()

    plt.show()

def train_lr(df: pd.DataFrame, feature: str, y_label: str, alpha: float = 0.001, max_iter: int = 100000) -> None:
    X = np.array(df[[feature]])
    Y = np.array(df[[y_label]])
    
    mean_x = mean(X)
    std_x = std(X)
    norm_params = np.array([mean_x, std_x])
    X = normalisation(X, norm_params)
    
    linear_model = MyLR(thetas=np.array([[0.], [0.]]), alpha=alpha, max_iter=max_iter)

    print(f"RMSE before -> {linear_model.rmse_(Y, linear_model.predict_(X))}")

    linear_model.fit_(X, Y)
    y_hat = linear_model.predict_(X)

    print(f"RMSE after -> {linear_model.rmse_(Y, y_hat)}")
    plot_lr(X, Y, y_hat, feature.lower(), colors=("darkblue", "dodgerblue"), lr_model=linear_model)

    theta_df = pd.DataFrame(linear_model.thetas, columns=['thetas'])

    theta_df.to_csv('thetas.csv', index=False)
    np.save('norm_params.npy', norm_params)


def main():
    try:
        df = pd.read_csv(sys.argv[1])
    except Exception as e:
        exit(e)
    train_lr(df, 'km', 'price', alpha=0.01, max_iter=1000)

if __name__=='__main__':
    main()
