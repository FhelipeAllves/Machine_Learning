# Import package
# import matplotlib as mpl
# mpl.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from linearRegCostFunction import linearRegCostFunction
from linearRegGradientFunction import linearRegGradientFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize
from plotFit import plotFit




# Machine Learning Online Class
#  Exercise 4 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.py
#     linearRegGradientFunction.py
#     learningCurve.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

# =========== Part : Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...')

# Load from ex4data1
data = scipy.io.loadmat('ex4data1.mat')
Xtrain = data['X']  # Dimensions de X, y, etc..????
ytrain = data['y']  #
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']

# m = Number of examples
m,n = Xtrain.shape

# Plot training data
# plt.figure()
plt.scatter(Xtrain, ytrain, marker='x', s=100, lw=1.5)
plt.ylabel('Water flowing out of the dam (y)')            # Set the y-axis label
plt.xlabel('Change in water level (x)')     # Set the x-axis label
plt.grid()
plt.show()
# plt.show(block=False)



# =========== Part : Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear
#  regression.
#

# initialisation: theta, Lambda
theta = np.array([[1, 1]]).T
Lambda = 1

# Add intercept
Xtrain_stack = np.column_stack((np.ones(m), Xtrain))

# compute cost for training set
J = linearRegCostFunction(Xtrain_stack, ytrain, theta, Lambda)

print('Cost at theta = [1  1]: %f \n(this value should be about 303.993192)\n' % J[0])





# =========== Part : Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear
#  regression.
#


# initialisation: theta, Lambda
theta = np.array([[1, 1]]).T
Lambda = 1

# Add intercept
Xtrain_stack = np.column_stack((np.ones(m), Xtrain))

# Cost
grad = linearRegGradientFunction(Xtrain_stack, ytrain, theta, Lambda)

print('Gradient at theta = [1  1]:  [%f %f] \n(this value should be about [-15.303016 598.250744])\n' %(grad[0], grad[1]))






# =========== Part : Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train
#  regularized linear regression.
#
#  Write Up Note: The data is non-linear, so this will not give a great
#                 fit.
#

#  Train linear regression with Lambda = 0
Lambda = 0
Xtrain_stack = np.column_stack((np.ones(m), Xtrain))
theta = trainLinearReg(Xtrain_stack, ytrain, Lambda)[0]

#  Prediction from the learned model
pred = Xtrain_stack.dot(theta)

#  Plot fit over the data
plt.figure()
plt.scatter(Xtrain, ytrain, marker='x', s=20, lw=1.5)
plt.ylabel('Water flowing out of the dam (y)')            # Set the y-axis label
plt.xlabel('Change in water level (x)')     # Set the x-axis label
plt.plot(Xtrain, pred, '--r', lw=2.0)
plt.grid()
plt.show()






# =========== Part : Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function.
#


Lambda = 0.
Xtrain_stack = np.column_stack((np.ones(m), Xtrain))
Xval_stack = np.column_stack((np.ones(Xval.shape[0]), Xval))


fig = plt.figure()
ax = fig.add_subplot(111)
cost_train, cost_val = learningCurve(Xtrain_stack, ytrain, Lambda, Xval_stack, yval, ax=ax)

plt.show()






# =========== Part : Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

maxiter = 3000

# set the max polynomial degree
p = 6

# Map X onto Polynomial Features and Normalize
X_poly_train = polyFeatures(Xtrain, p)

 # Normalize
X_poly_train, mu, sigma = featureNormalize(X_poly_train)

# Add Ones
X_poly_train = np.column_stack((np.ones(m), X_poly_train))


# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.column_stack((np.ones(X_poly_val.shape[0]), X_poly_val))           # Add Ones


# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.column_stack((np.ones(X_poly_test.shape[0]), X_poly_test))        # Add Ones





# =========== Part : Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of Lambda. The code below runs polynomial regression with
#  Lambda = 0. You should try running the code with different values of
#  Lambda to see how the fit and learning curve change.
#

Lambda = 0
theta, cost_train, cost_val = trainLinearReg(X_poly_train, ytrain, Lambda, X_poly_val, yval, maxiter=maxiter)


#  Plot fit over the training data
#fig = plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.scatter(Xtrain, ytrain, marker='x', s=60, lw=2)
plotFit(np.min(Xtrain), np.max(Xtrain), mu, sigma, theta, p, ax=ax1)
ax1.grid('on')
ax1.set_ylabel('Water flowing out of the dam (y)')            # Set the y-axis label
ax1.set_xlabel('Change in water level (x)')     # Set the x-axis label

# display learning curves
learningCurve(X_poly_train, ytrain, Lambda, X_poly_val, yval, ax=ax2, maxiter=maxiter)
ax2.grid('on')
ax2.set_ylabel('Costs')            # Set the y-axis label
ax2.set_xlabel('Iteration number')     # Set the x-axis label


plt.show()











# =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of
#  Lambda on a validation set. You will then use this to select the
#  "best" Lambda value.
#
# Train polynomial regression model
lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3, 10, 100])

cost_train_vs_lambda = np.zeros(lambda_vec.size)
cost_val_vs_lambda = np.zeros(lambda_vec.size)
for i, Lambda in enumerate(lambda_vec):

    # learn theta
    theta, cost_train, cost_val = trainLinearReg(X_poly_train, ytrain, lambda_vec[i], X_poly_val, yval, maxiter=maxiter)

    # Plot training data and fit
    #fig = plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(Xtrain, ytrain, marker='x', s=10, lw=2)
    plotFit(np.min(Xtrain), np.max(Xtrain), mu, sigma, theta, p, ax=ax1)

    ax1.set_xlabel('Change in water level (x)')            # Set the y-axis label
    ax1.set_ylabel('Water flowing out of the dam (y)')     # Set the x-axis label
    # plt.plot(X, np.column_stack((np.ones(m), X)).dot(theta), marker='_',  lw=2.0)
    ax1.set_title('Polynomial Regression Fit (Lambda = %f)' % Lambda)
    ax1.grid()


    # Plot Learning curves (Error vs Number of training examples)
    # error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, Lambda)
    # plt.figure()
    ax2.plot(range(cost_train.size), cost_train, color='b', lw=2, label='Train')
    ax2.plot(range(cost_val.size), cost_val, color='r', lw=2, label='Validation')
    ax2.set_title('Polynomial Regression Learning Curve (Lambda = %f)' % Lambda)
    ax2.set_xlabel('Number of iterations')
    ax2.set_ylabel('Cost')
    ax2.legend()
    ax2.grid()

    # for the final figure
    cost_train_vs_lambda[i] = cost_train[-1]
    cost_val_vs_lambda[i]   = cost_val[-1]

best_index = np.argmin(cost_val_vs_lambda)
best_lambda = lambda_vec[best_index]

print("Best lambda:", best_lambda)
print("Validation cost:", cost_val_vs_lambda[best_index])


# plot costs over lambdas
plt.figure()
plt.plot(lambda_vec, cost_train_vs_lambda, 'b', lw=2, label='train')
plt.plot(lambda_vec, cost_val_vs_lambda, 'r', lw=2, label='Validation')
plt.legend()
plt.xlabel('Lambda')
plt.ylabel('Error')
plt.grid()

plt.show()





# =========== Part:  Computing test set error =============
Lambda = 2.0 # change to Lambda to the best value

# Add intercept
theta = trainLinearReg(np.vstack((X_poly_train, X_poly_val)), np.vstack((ytrain, yval)), Lambda, maxiter=maxiter)[0]
# Cost
Jtest = linearRegCostFunction(X_poly_test, ytest, theta, 0)[0]
print('test set error: %f\n' % Jtest)



#  Plot fit over the data
plt.figure()
plt.scatter(Xtrain, ytrain, marker='x', s=20, lw=1.5)
plotFit(np.min(Xtrain), np.max(Xtrain), mu, sigma, theta, p, ax=plt.gca())

plt.ylabel('Water flowing out of the dam (y)')            # Set the y-axis label
plt.xlabel('Change in water level (x)')     # Set the x-axis label
plt.grid()
plt.show()


