
#%% package
import numpy as np
import matplotlib.pyplot as plt

from loadImages import loadImages
from selectFeatureVectors import selectFeatureVectors
from displayFeatures2d import displayFeatures2d
from displayFeatures3d import displayFeatures3d

#%% def preprocessing
def preprocessing():

    img73, img87 = loadImages()

    fig = plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.imshow(img73)
    plt.title("Aral Sea - 1973")

    plt.subplot(122)
    plt.imshow(img87)
    plt.title("Aral Sea - 1987")

    plt.show()

    margin = 50 
    img73 = img73[margin+10:-margin, :, :]
    img87 = img87[margin+10:-margin, :, :]

    fig = plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.imshow(img73)
    plt.title("Aral Sea - 1973")

    plt.subplot(122)
    plt.imshow(img87)
    plt.title("Aral Sea - 1987")

    # plt.show()

    featLearn, nbPix, nbFeat = selectFeatureVectors(img73, 500)

    print("Nombre de données d'apprentissage : ", nbPix)

    ax2D = displayFeatures2d(featLearn)
    ax3D = displayFeatures3d(featLearn)
    
    return featLearn,img73,img87

if __name__ == "__main__":
    preprocessing()