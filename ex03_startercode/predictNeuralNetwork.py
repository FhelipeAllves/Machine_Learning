import numpy as np

from sigmoid import sigmoid

def predictNeuralNetwork(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

    # Useful values
    m, _ = X.shape
    num_labels, _ = Theta2.shape

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a 
    #               vector containing labels between 1 to num_labels.
    #
    # Hint: The max function might come in useful. In particular, the np.argmax
    #       function can return the index of the max element, for more
    #       information see 'numpy.argmax' on the numpy website. If your examples 
    #       are in rows, then, you can use np.argmax(probs, axis=1) to obtain the 
    #       max for each row.
    #

    # =========================================================================
        
    z2 = X.dot(Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.column_stack((np.ones((m, 1)), a2)) # bias unit

    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)

    p = np.argmax(a3, axis=1)

    # =========================================================================
        
    return p

