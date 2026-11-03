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
# Question 6.2 / 7.2
# model = unsupervisedTraining(featLearn, method='kmeans')
model = unsupervisedTraining(featLearn, method='gmm')

# print("Centres des clusters:", model.cluster_centers_)
# print("Nombre d'iterations:", model.n_iter_)
# print("Inertie:", model.inertia_)
print("Moyennes des gaussiennes:")
print(model.means_)

print("Poids des gaussiennes:")
print(model.weights_)

print("Nombre d'itérations:", model.n_iter_)
print("Convergence:", model.converged_)
# ------------------------------------------------
# ------------------------------------------------

# prediction des labels sur la base d'apprentissage

# ------------------------------------------------
# YOUR CODE HERE
# Question 6.3
labelLearn = model.predict(featLearn)

print("Dimensions des labels:", labelLearn.shape)
print("Classes identifiées:", np.unique(labelLearn))
# ------------------------------------------------
# ------------------------------------------------

# Visualisation des resultats

# ------------------------------------------------
# YOUR CODE HERE
# Question 6.4
displayFeatures2d(featLearn, labelLearn)
displayFeatures3d(featLearn, labelLearn)

plt.show()
# ------------------------------------------------
# ------------------------------------------------

#%% Classement et estimation de la diminution de surface
# Classifying / Predicting / Testing

# mise en forme de l'image de 1973 et 1987 en matrice Num Pixels / Val Pixels

# ------------------------------------------------
# YOUR CODE HERE
# Question 6.5
nbPix73 = img73.shape[0] * img73.shape[1]
nbPix87 = img87.shape[0] * img87.shape[1]

# feat73 = img73.reshape((nbPix73, 3))
# feat87 = img87.reshape((nbPix87, 3))

# Question 6.5 - Mise en forme des images
feat73 = img73.reshape((nbPix73, 3)).astype(np.float64)
feat87 = img87.reshape((nbPix87, 3)).astype(np.float64)

print("Dimensions des descripteurs 1973:", feat73.shape)
print("Dimensions des descripteurs 1987:", feat87.shape)

label73 = unsupervisedClassifying(model, feat73)
label87 = unsupervisedClassifying(model, feat87)

print("Dimensions des labels 1973:", label73.shape)
print("Dimensions des labels 1987:", label87.shape)

print("Classes 1973:", np.unique(label73))
print("Classes 1987:", np.unique(label87))
# ------------------------------------------------
# ------------------------------------------------


# Classement des deux jeux de données et visualisation des résultats en image

# ------------------------------------------------
# YOUR CODE HERE
# Question 6.6
plt.figure()
imgLabel73, imgplot73 = displayImageLabel(label73, img73)
plt.title("Classification K-means - 1973")

plt.figure()
imgLabel87, imgplot87 = displayImageLabel(label87, img87)
plt.title("Classification K-means - 1987")

plt.show()
# ------------------------------------------------
# ------------------------------------------------



#%% Estimation de la surface perdue
answer = input('Numero de la classe de la mer ? ')
cl_mer = int(answer)

# ------------------------------------------------
# YOUR CODE HERE
# Question 6.7
surface73 = np.sum(label73 == cl_mer)
surface87 = np.sum(label87 == cl_mer)

diminution = (surface73 - surface87) / surface73 * 100

print("\n--- Estimation de la surface de la mer d'Aral ---")
print("Nombre de pixels en 1973:", surface73)
print("Nombre de pixels en 1987:", surface87)
print(f"Diminution de la surface: {diminution:.2f} %")









# ------------------------------------------------
# ------------------------------------------------



