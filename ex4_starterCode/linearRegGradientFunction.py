import numpy as np

def linearRegGradientFunction(X, y, theta, Lambda):
    """computes the gardients of the cost using theta as the parameter for linear regression
        to fit the data points in X and y.
        Returns the gradient in grad
     """

    # Initialize some values
    m,n = X.shape # number of training examples
    theta = theta.reshape((n,1)) # in case where theta is a vector (n,)
    grad = 0.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of regularized linear
    #               regression for a particular choice of theta.
    #
    #               You should set grad to the gradient.
    #

    h = X @ theta
    e = h - y.reshape((m,1))

    grad = (X.T @ e) / m
    grad[1:] += (Lambda / m) * theta[1:]

    # =========================================================================
    return grad.flatten()
