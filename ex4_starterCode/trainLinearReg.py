import numpy as np
from scipy.optimize import minimize

from linearRegCostFunction import linearRegCostFunction
from linearRegGradientFunction import linearRegGradientFunction


def trainLinearReg(X, y, Lambda, Xval=None, yval=None, maxiter=200):

    """trains linear regression using
    the dataset (X, y) and regularization parameter lambda. Returns the
    trained parameters theta.
    """


    # Initialize Theta and costs history
    theta = np.zeros(X.shape[1])
    n = theta.size  # number of parameters
    cost_train = []
    cost_val = []

    # fonction permettant à minimize de récupérer l'évolution des thetas
    thetas = []
    def save_step(*args):
        for arg in args:
            if type(arg) is np.ndarray:
                thetas.append(arg)

    # Initialize Theta
    initial_theta = np.zeros((X.shape[1]))

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda t: linearRegCostFunction(X, y, t, Lambda)
    gradFunction = lambda t: linearRegGradientFunction(X, y, t, Lambda)

    # Descente de gradient
    result = minimize(costFunction, initial_theta, method='CG', jac=gradFunction, options={'disp': False, 'maxiter': maxiter, 'return_all':True}, callback=save_step)


    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute
    #   - theta the final thetas
    #   - the training cost vs iteration number from thetas
    #   - the validation cost vs iteration number from thetas

    # outputs
    theta = result.x
    cost_train = [linearRegCostFunction(X, y, t, 0) for t in thetas]

    if Xval is not None and yval is not None:
        cost_val = [linearRegCostFunction(Xval, yval, t, 0) for t in thetas]




    # =========================================================================

    # force to np array
    cost_train = np.array(cost_train)
    cost_val = np.array(cost_val)

    return theta, cost_train, cost_val
