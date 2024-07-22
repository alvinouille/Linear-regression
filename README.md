# How to do a linear regression

## Supervised Learning:

1. **Dataset** :contains the target variable y and the feature variables x.
2. **Model**: the machine learning algorithm with parameters (coefficients) that the machine needs to learn: `f(x) = ax + b`.
3. **Cost function**: the measure of the errors between the model's predictions and the true values y of the dataset.
4. **Minimization algorithm**: aims to minimize the cost function by finding the optimal parameters.

## Linear Regression:

1. **Dataset**: *(x, y)* with *m* = 6 data points and *n* = 1 feature.
2. **Model**: *Linear model*: `f(x) = ax + b`.
    - The parameters are initially unknown, so we assign random values, e.g., 0, 0.
3. **Cost function**: dependent on a and b (as we can only change the parameters, not the model!):
    - To calculate the error: `(f(x) - y)^2`, which represents the Euclidean distance between two points.
    - *MSE (Mean Squared Error)* = average of the errors: `J(a, b) = sum(f(x) - y)^2 / 2m`.
        - Why 2m instead of m? We introduce a coefficient of `1/2` to simplify calculations without changing the direction of the results.
4. **Minimization function**:
    - *Gradient Descent*: an optimization algorithm that converges towards the minimum of a convex function (i.e., minimizing the cost function) to make the machine "learn."
    - How it works:
        - We start with a randomly chosen parameter a0 and calculate the error using the cost function.
        - In a loop:
            - Calculate the slope (if `J(a, b)` increases with increasing a or decreases with increasing a): compute the partial derivative of the cost function in function of a.
            - Take a small step in the descending direction indicated by the slope.
            Example, aiming to converge towards the minimum:
                - If the slope descends (`J(a, b)` decreases with increasing a), take a small step alpha in the direction of the slope (increase a).
                - If the slope increases (`J(a, b)` increases with increasing a), take a small step alpha in the opposite direction of the slope (decrease a).
            → New position a1.
        Continue this process until convergence to the minimum. Repeat the same steps for parameter b.
    - Formula: `a(i + 1) = a(i) - alpha * (derivative of J(a(i)))`
        - If the derivative of `J(a(i)) < 0`, `a(i + 1)` increases.
        - If the derivative of `J(a(i)) > 0`, `a(i - 1)` decreases.
        - The learning rate (alpha) is the product of the learning rate and the derivative.
    - Choosing the value of alpha:
        - If it is too large, it results in large steps and oscillation around the objective without reaching it.
        - If it is too small, it takes an infinite amount of time to reach the objective, resulting in NaN parameters.
        - The optimal value is found through trial and error.