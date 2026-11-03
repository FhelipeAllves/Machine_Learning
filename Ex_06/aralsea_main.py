#%% Machine Learning Class - Exercise Aral Sea Surface Estimation

# package
import numpy as np
import matplotlib.pyplot as plt

plt.ioff() # to see figure avant input


# ------------------------------------------------
# YOUR CODE HERE
from preprocessing import preprocessing
from unsupervisedTraining import unsupervisedTraining
from displayFeatures2d import displayFeatures2d
from displayFeatures3d import displayFeatures3d
from unsupervisedClassifying import unsupervisedClassifying
from displayImageLabel import displayImageLabel

# ------------------------------------------------
# ------------------------------------------------





#%% Examen des données, prétraitements et extraction des descripteurs

# Chargement des données et prétraitements
featLearn,img73,img87 = preprocessing()




#%% Apprentissage / Learning / Training


# Apprentissage de la fonction de classement

# ------------------------------------------------
# YOUR CODE HERE

model = unsupervisedTraining(featLearn, method='kmeans')

# ------------------------------------------------
# ------------------------------------------------




# prediction des labels sur la base d'apprentissage

# ------------------------------------------------
# YOUR CODE HERE

labelsLearn = model.predict(featLearn)

# ------------------------------------------------
# ------------------------------------------------




# Visualisation des resultats

# ------------------------------------------------
# YOUR CODE HERE

ax2D = displayFeatures2d(featLearn, labelsLearn)
ax3D = displayFeatures3d(featLearn, labelsLearn)

# ------------------------------------------------
# ------------------------------------------------



#%% Classement et estimation de la diminution de surface
# Classifying / Predicting / Testing


# mise en forme de l'image de 1973 et 1987 en matrice Num Pixels / Val Pixels

# ------------------------------------------------
# YOUR CODE HERE

h73, w73, c73 = img73.shape 
h87, w87, c87 = img87.shape

feat73 = img73.reshape(h73*w73, c73)
feat87 = img87.reshape(h87*w87, c87)

# ------------------------------------------------
# ------------------------------------------------


# Classement des deux jeux de données et visualisation des résultats en image

# ------------------------------------------------
# YOUR CODE HERE

labels73 = unsupervisedClassifying(model, feat73)
labels87 = unsupervisedClassifying(model, feat87)

plt.figure()
plt.title("Classification 1973")
imglabel73, imgplot73 = displayImageLabel(labels73, img73)

# Affichage pour 1987 (Correction de l'erreur labels73/img73)
plt.figure()
plt.title("Classification 1987")
imglabel87, imgplot87 = displayImageLabel(labels87, img87)

plt.show()
# ------------------------------------------------
# ------------------------------------------------



#%% Estimation de la surface perdue
answer = input('Numero de la classe de la mer ? ')
cl_mer = int(answer)

surf73 = np.sum(labels73 == cl_mer)
surf87 = np.sum(labels87 == cl_mer)

print(f"Surface (en pixels) en 1973 : {surf73}")
print(f"Surface (en pixels) en 1987 : {surf87}")


evolution = ((surf87 - surf73) / surf73) * 100

print(f"Évolution de la surface de la mer d'Aral : {evolution:.2f} %")











# ------------------------------------------------
# ------------------------------------------------



