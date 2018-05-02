
# coding: utf-8

# Implémentation de K-means clustering python
# Code produit par le site https://mrmint.fr

#Chargement des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets


#chargement de jeu des données Iris
iris = datasets.load_iris()

#importer le jeu de données Iris dataset à l'aide du module pandas
x = pd.DataFrame(iris.data)

x.columns = ['Sepal_Length','Sepal_width','Petal_Length','Petal_width']


y = pd.DataFrame(iris.target)


y.columns = ['Targets']


#Création d'un objet K-Means avec un regroupement en 3 clusters (groupes)
model=KMeans(n_clusters=3)



#application du modèle sur notre jeu de données Iris
model.fit(x)



#Visualisation des clusters
plt.scatter(x.Petal_Length, x.Petal_width)
plt.show()




colormap=np.array(['Red','green','blue'])



#Visualisation du jeu de données sans altération de ce dernier (affichage des fleurs selon leur étiquettes)
plt.scatter(x.Petal_Length, x.Petal_width,c=colormap[y.Targets],s=40)
plt.title('Classification réelle')
plt.show()

#Visualisation des clusters formés par K-Means
plt.scatter(x.Petal_Length, x.Petal_width,c=colormap[model.labels_],s=40)
plt.title('Classification K-means ')
plt.show()


