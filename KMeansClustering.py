from sklearn import datasets, metrics
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()

iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names']+['species'])

print("Data awal iris : ")
print(iris)

iris.columns = iris.columns.str.replace(' ', '')
iris.head()

X = iris.iloc[:,:3]
y = iris.species
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

model = KMeans(n_clusters=3, random_state=11)
model.fit(X)
print("Label hasil Clustering KMeans adalah :")
print(model.labels_)

iris['pred_species'] = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)

print("Accuracy :", metrics.accuracy_score(iris.species, iris.pred_species))
print("Classification report :", metrics.classification_report(iris.species, iris.pred_species))

print("Cetak data iris dengan full rows :")
pd.set_option('display.max_rows', None)
print(iris)

print("Cetak data iris dengan sampel acak sebesar 0,3 :")
print(iris.sample(frac=0.3))

