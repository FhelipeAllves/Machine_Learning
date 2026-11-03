#%% package
import numpy as np
from sklearn.cluster import KMeans


#%% def unsupervisedClassifying
def unsupervisedClassifying(model, feat):
    ''' classement/prédiction à partir d'un modèle de classement non supervisé
    feat est la matrice du jeu de données à classer
    label est la classe prédite
    '''
	
	# ------------------------------------------------
	# YOUR CODE HERE
    feat = np.asarray(feat, dtype=np.float64)
    label = model.predict(feat)
	# ------------------------------------------------
	# sortie
    return label




