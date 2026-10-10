import numpy as np 

y = np.array([[2, 0, 1, 3, 2, 0, 1, 3, 0, 0]]).T

y_train = np.zeros((10, 4))

y_train[np.arange(10), y.flatten()] = 1

print(y_train)
