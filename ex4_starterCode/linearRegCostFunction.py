import numpy as np
def linearRegCostFunction(X, y, theta, Lambda):
    """computes the cost of using theta as the parameter for linear regression
    to fit the data points in X and y.
    Returns the cost in J
    """

    # Initialize some useful values

    m,n = X.shape # number of training examples
    theta = theta.reshape((n,1)) # in case where theta is a vector (n,) 
    J = 0.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of regularized linear
    #               regression for a particular choice of theta.
    #
    #               You should set J to the cost.
    #

    h = X.dot(theta) 
    a = (1/(2*m)) * np.sum(np.square(h - y))
    b = (Lambda/(2*m)) * np.sum(np.square(theta[1:]))
    J = a+b

    # =========================================================================
    return J.flatten()







