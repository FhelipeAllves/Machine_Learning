import matplotlib.pyplot as plt
from trainLinearReg import trainLinearReg

from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction
from linearRegGradientFunction import linearRegGradientFunction

def learningCurve(X, y, Lambda, Xval, yval, ax=plt.gca(), maxiter=2000):
    """returns the train and validation/test set costs for a learning curve. In particular,
    it returns two vectors of the same length - cost_train and
    cost_val.

    In this function, you will compute the train and validation costs.
    """

    # Number of training examples
    m, _ = X.shape
    cost_train = 0.
    cost_val = 0.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return training costs in
    #               cost_train and the validation errors in cost_val.
    #

    _, cost_train, cost_val =trainLinearReg(X, y, Lambda, Xval, yval, maxiter=maxiter)

    # Erreur d'apprentissage et de généralisation

    # =========================================================================

    # display
    plt.figure()
    ax.plot(range(cost_train.size), cost_train, color='b', lw=2, label='Train')
    ax.plot(range(cost_val.size), cost_val, color='r', lw=2, label='Validation')
    ax.set_title('Learning curves for linear regression')
    ax.grid()
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Cost')
    ax.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)

    return cost_train, cost_val