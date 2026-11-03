#%% package
import numpy as np
import matplotlib.pyplot as plt

from loadImages import loadImages
from selectFeatureVectors import selectFeatureVectors
from displayFeatures2d import displayFeatures2d
from displayFeatures3d import displayFeatures3d


#%% def preprocessing
def preprocessing():
    # ------------------------------------------------
    # YOUR CODE HERE
    # Question 3.1
    img73, img87 = loadImages()

    # Question 3.4
    img73 = img73[75:840, :, :]
    img87 = img87[75:840, :, :]

    # Question 4.1
    featLearn, nbPix, nbFeat = selectFeatureVectors(img73, 500)

    # Question 5.1
    displayFeatures2d(featLearn)
    displayFeatures3d(featLearn)

    plt.show()

    plt.figure()
    plt.imshow(img73)
    plt.title("Mer d'Aral (1973)")

    plt.figure()
    plt.imshow(img87)
    plt.title("Mer d'Aral (1987)")

    plt.show()

    print("Dimensions 1973:", img73.shape)
    print("Dimensions 1987:", img87.shape)

    print("\nNombre de données d'apprentissage:", nbPix)
    print("Nombre de descripteurs:", nbFeat)
    print("Dimensions de featLearn:", featLearn.shape)
    # ------------------------------------------------
    # ------------------------------------------------
    #%% sortie
    return featLearn,img73,img87
    