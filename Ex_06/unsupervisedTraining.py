#%% package
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

#%% def unsupervisedTraining
def unsupervisedTraining(featLearn, method='kmeans'):
    ''' apprentissage avec la fonction KMeans() et GaussianMixture 
    de scikit-learn :
    - featLearn est la matrice de l'ensemble d'apprentissage
    - method: type d'algorithme de Machine Learning utilisé (KMeans et GaussianMixture) 
    - nbCluster est le nombre de cluster = nombre de classes rentré par l'utilisateur en début de fonction

    - renvoie model: le modèle de classement ou classifieur
    '''


    # fixer le nombre de classes
    answer = input('nombre de classes:')
    nbCluster = int(answer)

    if method == 'kmeans':

        # Xs= StandardScaler().fit_transform(featLearn) #normaliserlesdescripteurs
        Xs = featLearn
        #---K-means------------------------------------------------------
        km= KMeans(n_clusters=nbCluster,init="k-means++",n_init=10, random_state=0).fit(Xs)
        model = km
        labels_km =km.labels_ #affectationdure
        J = km.inertia_ #fonctiondecoutJ
        s = silhouette_score(Xs,labels_km) #qualiteinterne


    elif method == 'gmm':
        model = None


    return model

# %%
