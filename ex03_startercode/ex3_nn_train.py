#%% Machine Learning Online Class - Exercise 4 Neural Network Learning

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoidGradient.py
#     randInitializeWeights.py
#     nnCostFunction.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import numpy as np
import scipy.io
from scipy.optimize import fmin_cg

from displayData import displayData
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients
from predictNeuralNetwork import predictNeuralNetwork

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

#%% =========== Part : Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('\n -------------------------- \n')
print('Loading and Visualizing Data ...')

data = np.load('ex3data.npz')
X = data['X']
y = data['y']
m, _ = X.shape

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100], :]

displayData(sel)



#%% ================ Part : Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\n -------------------------- \n')
print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
data = np.load('ex3weights.npz')
Theta1 = data['Theta1'].astype('float64')
Theta2 = data['Theta2'].astype('float64')

# Unroll parameters 
nn_params = np.hstack((Theta1.T.ravel(), Theta2.T.ravel()))




#%% ================ Part : Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.py to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('\n -------------------------- \n')
print('Feedforward Using Neural Network ...')

# Weight regularization parameter (we set this to 0 here).
Lambda = 0

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
    num_labels, X, y, Lambda)

print('\n -------------------------- \n')
print('Cost at parameters (loaded from ex3weights): %f \n(this value should be about 0.287629)\n' % J)




#%% =============== Part : Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('\n -------------------------- \n')
print('Checking Cost Function (w/ Regularization) ...')

# Weight regularization parameter (we set this to 1 here).
Lambda = 1

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

print('\n -------------------------- \n')
print('Cost at parameters (loaded from ex3weights): %f \n(this value should be about 0.383770)' % J)



#%% ================ Part : Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.py file.
#

print('\n -------------------------- \n')
print('Evaluating sigmoid gradient...')

g = sigmoidGradient(np.array([1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:' + str(g))
print('(this value should be about 0.196612 0.235004 0.250000 0.235004 0.196612)')
  




#%% ================ Part : Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.py)

print('\n -------------------------- \n')
print('Initializing Neural Network Parameters ...')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.hstack((initial_Theta1.T.ravel(), initial_Theta2.T.ravel()))







#%% =============== Part : Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.py to return the partial
#  derivatives of the parameters.
#
print('\n -------------------------- \n')
print('Checking Backpropagation... ')

#  Check gradients by running checkNNGradients
checkNNGradients()





#%% =============== Part : Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('\n -------------------------- \n')
print('Checking Backpropagation (w/ Regularization) ... ')

#  Check gradients by running checkNNGradients
Lambda = 3.0
checkNNGradients(Lambda)

# Also output the costFunction debugging values
debug_J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

print('Cost at (fixed) debugging parameters (w/ lambda = 10): %f (this value should be about 0.576051)\n\n' % debug_J)



#%% =================== Part : Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmin_cg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('\n -------------------------- \n')
print('Training Neural Network... ')

#  After you have completed the assignment, change the MaxIter to a larger
#  value to see how more training helps.

#  You should also try different values of lambda
Lambda = 1

costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)[0]
gradFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)[1]

result = fmin_cg(costFunc, fprime=gradFunc, x0=initial_nn_params, maxiter=50, disp=True,full_output=True)
nn_params = result[0]
cost = result[1]


# Obtain Theta1 and Theta2 back from nn_params
Theta1 = nn_params[0:(hidden_layer_size*(input_layer_size+1))].reshape((input_layer_size+1),hidden_layer_size).T
Theta2 = nn_params[(hidden_layer_size*(input_layer_size+1)):].reshape((hidden_layer_size+1),num_labels).T






#%% ================= Part : Visualize Weights =================
#  You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
#  the data.

print('\n -------------------------- \n')
print('Visualizing Neural Network... ')

displayData(Theta1[:, 1:])





#%% ================= Part : Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.
X = np.column_stack((np.ones((m, 1)), X))
pred = predictNeuralNetwork(Theta1, Theta2, X)

accuracy = np.mean(np.double(pred == y.flatten())) * 100
print('Training Set Accuracy: %f\n'% accuracy)


