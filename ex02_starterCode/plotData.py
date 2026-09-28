import numpy as np
from matplotlib import pyplot as plt


def plotData(X,y):


	# ====================== YOUR CODE HERE ======================
	# Instructions: Plot data X with different markers according the value in y
	# =============================================================

	pos = X[(y==1).flatten(),:]
	neg = X[(y==0).flatten(),:]
	plt.figure()
	plt.scatter(neg[:, 0], neg[:, 1], s=20, marker='P', color='b')
	plt.scatter(pos[:, 0], pos[:, 1], s=20, marker='o', color='r')
	plt.xlabel('Exam 1 score')
	plt.ylabel('Exam 2 score')
	plt.grid(True)
	plt.legend(['Not admitted (y=0)','Admitted (y=1)'], loc='upper right', shadow=True,fontsize='x-large', numpoints=1)

	# =============================================================
