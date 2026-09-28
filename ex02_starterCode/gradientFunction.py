from sigmoid import sigmoid
import numpy as np


def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic 
    regression and the gradient of the cost w.r.t. to the parameters.
    """

    # Initialize some useful values
    # number of training examples 
    m = X.shape[0]   

    # number of parameters
    n = X.shape[1]   
    theta = theta.reshape((n,1)) # due to the use of fmin_tnc


    # gradient variable
    grad = 0.


    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of a particular choice of theta.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    arg1 = X @ theta
    arg2 = sigmoid(arg1)
    grad = (X.T @ (arg2 - y)) / m


    # =============================================================

    return grad