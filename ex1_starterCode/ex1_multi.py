# ================ Introduction: load packages ================
# global

import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt

from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn
from predict_multi import predict_multi


#%% ================ Part : Feature Normalization ================

print('\n -------------------------- \n')
print('Loading data ...')

# Load Data
path ='ex1data2.txt'
data = np.loadtxt(path, delimiter=',', dtype=float)

# set X (training data) and y (target variable)
nbCol = data.shape[1]
X = data[:, 0:nbCol-1]
y = data[:,nbCol-1:nbCol]

m = X.shape[0]

# Plot some data points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# X[:, 0] -> first column of X, X[:, 1] -> second column of X, y -> y values
ax.scatter(X[:, 0], X[:, 1], y[:,0], c='r', marker='o', s=80)
ax.set_xlabel('Surface area [m²]')
ax.set_ylabel('Bedroom []')
ax.set_zlabel('Price [$]')
# plt.show()


# Print out some data points
print('\n -------------------------- \n')
print('First 10 examples from the dataset:')
print(np.column_stack( (X[:10], y[:10]) ))














# Scale features and set them to zero mean
print('\n -------------------------- \n')
print('Normalizing Features ...')

X, mu, sigma = featureNormalize(X)
print('[mu] [sigma]')
print(mu, sigma)

# Add intercept term to X
X = np.concatenate((np.ones((m, 1)), X), axis=1)


#%% ================ Part : Gradient Descent ================
#
# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha).
#
#               Your task is to first make sure that your functions -
#               computeCost and gradientDescent already work with
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
#
# Hint: At prediction, make sure you do the same feature normalization.
#

print('\n -------------------------- \n')
print('Running gradient descent ...')

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent 
n = X.shape[1]
theta = np.zeros((n,1))
theta, cost_history, theta_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Display gradient descent's result
print('\n -------------------------- \n')
print('Theta computed from gradient descent: ')
print(theta)



# Plot the convergence graph
fig = plt.figure()
ax = plt.gca()
ax.plot(np.arange(num_iters), cost_history, color="blue", linewidth=2.0, linestyle="-")  
ax.grid()
ax.set_xlabel('iteration number')  
ax.set_ylabel(r'Cost J($\theta$)')  
ax.set_title('Error vs. Training Epoch (number of iters)')  
# plt.show()



# ================ Part : Selecting learning rates  ================
alpha = [0.001, 0.01, 0.03,0.1, 0.3, 1.0]
fig = plt.figure()
col = 'brkg'
for i, alp in enumerate(alpha):
    print('\nTesting alpha = ', alp)
    theta = np.zeros((n, 1))
    theta, cost_history, theta_history = gradientDescentMulti(X, y, theta, alp, num_iters)

    # Plot the convergence graph
    plt.plot(np.arange(num_iters), cost_history, linewidth=2.0, label=f'alpha = {alp}')

plt.grid()
plt.legend()
plt.xlabel('iteration number')
plt.ylabel(r'Cost J($\theta$)')
plt.title('Error vs. Training Epoch (number of iters)')
# plt.show()









# ================ Part : Predict the price for a new house ================
alpha = 1.0
num_iters = 400

# Init Theta and Run Gradient Descent
n = X.shape[1]
theta = np.zeros((n,1))
theta, cost_history, theta_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

new_house = np.array([[1650, 3]])
price = predict_multi(new_house, mu, sigma, theta)


print('\n -------------------------- \n')
print('Predicted price of a 1650 sq-ft, 3 br house')
print('(using gradient descent): ')
print(price)


# PLot to visualize the plan created by the function theta
t = np.linspace(X[:,1].min(), X[:,1].max(), 100)
tt = np.linspace(X[:,2].min(), X[:,2].max(), 100)
bd = np.meshgrid(t, tt)
h = theta[0] + theta[1]*bd[0] + theta[2]*bd[1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 1], X[:, 2], y[:,0], c='r', marker='o', s=80)
ax.plot_surface(bd[0], bd[1], h, alpha=0.5, color='blue')
ax.set_xlabel('Surface area [m²]')
ax.set_ylabel('Bedroom []')
ax.set_zlabel('Price [$]')
ax.set_title(' PLot to visualize the plan created by the function theta')
plt.show()





#%% ================ Part 3: Normal Equations ================

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form
#               solution for linear regression using the normal
#               equations. You should complete the code in
#               normalEqn.py
#
#               After doing so, you should complete this code
#               to predict the price of a 1650 sq-ft, 3 br house.
#
print('\n -------------------------- \n')
print('Solving with normal equations...')

# Load Data
path = 'ex1data2.txt'
data = np.loadtxt(path, delimiter=',')


# set X (training data) and y (target variable)
nbCol = data.shape[1]  
X = data[:,0:nbCol-1]
y = data[:,nbCol-1:nbCol]
m = X.shape[0]



# Add intercept term to X
X = np.concatenate((np.ones((m, 1)), X), axis=1)

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations:')
print(' %s \n' % theta)

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.array([[1, 1650, 3 ]]).dot(theta)


print("Predicted price of a 1650 sq-ft, 3 br house ")
print('(using normal equations):\n $%f\n' % price.flatten())

# ============================================================
