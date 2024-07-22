import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR
from utils import normalisation

def get_valid_mileage():
    while True:
        try:
            mileage = float(input("Enter a mileage : "))
            if 0 <= mileage <= 300000:
                return mileage
            else:
                print("Invalid input. Please enter a value between 0 and 300 000.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

def predict_(thetas, norm_params) -> None:
    linear_model = MyLR(thetas=thetas)
    X = np.array([[get_valid_mileage()]])

    if norm_params is not None:
        X = normalisation(X, norm_params)

    y_hat = linear_model.predict_(X)
    print(f"Estimated price in fonction of this mileage : {y_hat[0, 0]}")


def main():
    try:
        thetas = pd.read_csv('thetas.csv').to_numpy()
    except Exception as e:
        thetas = np.array([[0.], [0.]])
        print(f"Be careful, thetas initialized to [[0.], [0.]] because : {e}")
    try:
        norm_params = np.load('norm_params.npy')
    except Exception as e:
        print(f"Be careful, data has not been normalized because {e}")
        norm_params = None
    predict_(thetas, norm_params)

if __name__=='__main__':
    main()