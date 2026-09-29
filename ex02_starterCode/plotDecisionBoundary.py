import numpy as np
from matplotlib import pyplot as plt
from plotData import plotData
from matplotlib.colors import CenteredNorm, Normalize

def plotDecisionBoundary(theta, X, y, Lambda):
    """
    Plots the data points X and y into a new figure with the decision boundary 
    defined by theta     
      PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
      positive examples and o for the negative examples. X is assumed to be
      a either
      1) Mx3 matrix, where the first column is an all-ones column for the
         intercept.
      2) MxN, N>3 matrix, where the first column is all-ones
    """

    # Plot Data
    # plt.figure()
    plotData(X[:,1:], y)

    if X.shape[1] <= 3:

        # plot decision boundary line
        plot_x = np.array([min(X[:, 1]),  max(X[:, 1])])
        plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])
        plt.plot(plot_x, plot_y, 'k', lw=5)

        # cpt model outputs before sigmoid
        x1 = np.linspace(0, 100, 100)
        x2 = np.linspace(0, 100, 100)
        XX1, XX2 = np.meshgrid(x1, x2)
        S = np.array([np.ones(np.size(XX1)), XX1.flatten(), XX2.flatten()]).T @ np.atleast_2d(theta).T
        S = S.reshape(XX1.shape)

        # plot model outputs before sigmoid
        cmap = 'bwr'  # 'RdBu'
        cmap = plt.get_cmap(cmap)
        plt.imshow(S, cmap=cmap, norm=CenteredNorm(0)   )
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.axis([X[:, 1].min(), X[:, 1].max(), X[:, 2].min(), X[:, 2].max()])



    else:

        # cpt model outputs before sigmoid
        xvals = np.linspace(-1,1.5,100)
        yvals = np.linspace(-1,1.5,100)
        zvals = np.zeros((len(xvals),len(yvals)))
        for i in range(len(xvals)):
            for j in range(len(yvals)):
                myfeaturesij = mapFeature(np.array([xvals[i]]),np.array([yvals[j]]))
                zvals[i][j] = np.dot(theta.flatten(),myfeaturesij.T)
        zvals = zvals.transpose()

        # plot model outputs before sigmoid
        cmap = 'bwr'  # 'RdBu'
        cmap = plt.get_cmap(cmap)
        plt.imshow(zvals, cmap=cmap, norm=Normalize(vmin=-10, vmax=10, clip=True), alpha=0.5, extent=[-1, 1.5, -1, 1.5],
                   origin='lower', interpolation='bilinear')
        plt.colorbar()

        # plot decision boundary curve
        mycontour = plt.contour(xvals, yvals, zvals, [0], colors='g')
        # Kind of a hacky way to display a text on top of the decision boundary
        myfmt = {0: 'Lambda = %d' % Lambda}
        plt.clabel(mycontour, inline=1, fontsize=15, fmt=myfmt)
        plt.title("Decision Boundary")
    plt.savefig(f'rapport/figure/boundary_{Lambda}.png', dpi=300, bbox_inches='tight')
    plt.show()

def mapFeature(x1col, x2col, degree=6):
    """
    Feature mapping function to polynomial features

    MAPFEATURE(X, degree) maps the two input features
    to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    """
    """ 
    Function that takes in a column of n- x1's, a column of n- x2's, and builds
    a n- x 28-dim matrix of features as described in the assignment
    """
    out = np.ones( (x1col.shape[0], 1) )

    for i in range(1, degree+1):
        for j in range(0, i+1):
            term1 = x1col ** (i-j)
            term2 = x2col ** (j)
            term  = (term1 * term2).reshape( term1.shape[0], 1 ) 
            out   = np.hstack(( out, term ))
    return out    

        
