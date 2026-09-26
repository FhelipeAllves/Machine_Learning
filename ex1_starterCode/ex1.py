# ================ Introduction: load packages ================

# global
import numpy as np
import matplotlib as mpl
# mpl.use('TkAgg')  # or can use 'Qt5Agg', whatever you have/prefer
import matplotlib.pyplot as plt

# local
from computeCost import computeCost
from gradientDescent import gradientDescent
from plotData import plotData

# ================                             ================
# Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following modules
#  in this exericse:
#
#     plotData.py
#     gradientDescent.py
#     computeCost.py
#     gradientDescentMulti.py
#     computeCostMulti.py
#     featureNormalize.py
#     normalEqn.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s





#%% ======================= Part: Plotting =======================

# Read data using numpy
path = 'ex1data1.txt'
data = np.loadtxt(path, delimiter=',', dtype=float)


# set X (training data) and y (target variable)
nbCol = data.shape[1]  
X = data[:, 0:nbCol-1]
y = data[:,nbCol-1:nbCol]

# Plot Data
# Note: You have to complete the code in plotData.py
# plotData(X,y)







#%% =================== Part: Gradient descent ===================
m = X.shape[0]

# Add intercept term to X
#X = np.concatenate((np.ones((m, 1)), X), axis=1)
X = np.column_stack((np.ones((m, 1)), X)) #works fine too

# initialize theta: shape = (2,1)
theta = np.array([[0.,0.]]).T


# compute and display initial cost
# Note: You have to complete the code in computeCost.py
J = computeCost(X, y, theta)
print('\n -------------------------- \n')
print('cost: %0.4f ' % J)
print('Expected cost value (approx) 32.07')

# further testing of the cost function
J = computeCost(X, y, np.array([[-1, 2]]).T)
print('\n -------------------------- \n')
print('With theta = [-1 ; 2] Cost computed = %f' %J)
print('Expected cost value (approx) 54.24')



# compute Descent gradient
# initialize variables for learning rate and iterations
alpha = 0.01
iters = 1500

# perform gradient descent to "fit" the model parameters
# Note: You have to complete the code in gradientDescent.py
theta, cost_history, theta_history = gradientDescent(X, y, theta, alpha, iters)  


# print theta to screen
print('\n -------------------------- \n')
print('Theta found by gradient descent: ')
print('%s %s' % (theta[0,0], theta[1,0]))
print('Expected theta values (approx)')
print(' -3.6303  1.1664')






# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([[1, 3.5]]).dot(theta)
predict2 = np.array([[1, 7]]).dot(theta)
#predict1 = np.array([[1, 3.5]])@theta
#predict2 = np.array([[1, 7]])@theta

print('\n -------------------------- \n')
print('For population = 35,000, we predict a profit of {:.4f}'.format(predict1[0,0]*10000))
print('For population = 70,000, we predict a profit of {:.4f}'.format(predict2[0,0]*10000))






# Checking the goodness-of-fit
# Fit=estimation de la droite de régression
x = np.linspace(X.min(), X.max(), 100)
f = theta[0, 0] + (theta[1, 0] * x)


# Plot the linear fit
fig = plt.figure(figsize=(12,8))  
ax = plt.gca()
ax.plot(x, f, 'r', label='Linear regression: h(x) = %0.2f + %0.2fx'%(theta[0,0],theta[1,0]))  
ax.scatter(X[:,1], y, label='Training Data')
ax.legend(loc=2)  
ax.set_xlabel('Population')  
ax.set_ylabel('Profit')  
ax.set_title('Predicted Profit vs. Population Size')  
ax.grid()
#plt.show()








#%% =================== Part: Visualizing J(theta)= J(theta_0, theta_1) ===================

# Checking the convergence =  Evolution du coût
fig= plt.figure(figsize=(12,8))
ax = plt.gca()
ax.plot(np.arange(iters), cost_history, color="blue", linewidth=2.0, linestyle="-")
ax.set_xlabel('iteration number')
ax.set_ylabel(r'Cost J($\theta$)')
ax.set_title('Error vs. Training Epoch (number of iters)')
ax.grid()
ax.set_xlim([-20,1600])
ax.set_ylim([4,7])


print('\n -------------------------- \n')
print('Visualizing J(theta_0, theta_1) ...')

# Create grid coordinates for plotting
theta0 = np.linspace(-10, 10, 100)
theta1 = np.linspace(-1, 4, 100)
theta0, theta1 = np.meshgrid(theta0, theta1, indexing='xy')

Z = np.zeros((theta0.shape[0],theta1.shape[0]))

# Calculate Z-values (Cost) based on grid of coefficients
for (i,j),v in np.ndenumerate(Z):
    t = np.array([[theta0[i,j], theta1[i,j]]]).T
    Z[i,j] = computeCost(X,y, t)








# Créer une colormap personnalisée avec 100 niveaux
colors = [(1, 0.8, 0.8), (1, 0, 0)]  # Dégradé du rouge pâle (RGB: 1, 0.8, 0.8) au rouge vif (RGB: 1, 0, 0)
cmap_name = 'red_gradient'
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=theta_history.shape[1])


fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133, projection='3d')

# Left plot
ax1.scatter(X[:,1], y, s=15, label='Training Data')
ax1.set_xlabel('Population')
ax1.set_ylabel('Profit')
ax1.set_title('Predicted Profit vs. Population Size')
ax1.grid()

# mid plot
CS = ax2.contour(theta0, theta1, Z, np.geomspace(Z.min(),Z.max(),10), cmap=plt.cm.jet, color='black')
plt.clabel(CS, inline=1, fontsize=10)
ax2.scatter(theta_history[0,:],theta_history[1,:], c='r')
ax2.grid()

# Right plot
ax3.plot_surface(theta0, theta1, Z, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet, linewidth=0, antialiased=True)
ax3.set_zlabel('Cost')
ax3.set_zlim(Z.min(),Z.max())
ax3.view_init(elev=15, azim=230)
ax3.grid()


x = np.linspace(X.min(), X.max(), 100)
for idx in np.arange(theta_history.shape[1]):

    f = theta_history[0, idx] + (theta_history[1, idx] * x)
    ax1.plot(x, f, color=cmap(idx/theta_history.shape[1]), linewidth=2)

    ax2.scatter(theta_history[0,idx],theta_history[1,idx],  color=cmap(idx/theta_history.shape[1]))

    ax3.scatter(theta_history[0,idx],theta_history[1,idx], cost_history[idx],  color=cmap(idx/theta_history.shape[1]))


plt.show()
