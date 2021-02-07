from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()
iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['species'])

iris.columns = iris.columns.str.replace(' ','')
iris.head()

X = iris.iloc[:, :3]
y = iris.species
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

model = KMeans(n_clusters=3, random_state=11)
model.fit(X)

iris['pred_species'] = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)

fig, ax1 = plt.subplots(2, 2, figsize=(22, 18), gridspec_kw={'hspace': 0.5, 'wspace': 0.2})

colorplot = dict({0.0:'red', 0:'red', 1.0:'green', 1:'green', 2.0:'blue', 2:'blue'})

sns.scatterplot(data=iris, x='sepallength(cm)', y='sepalwidth(cm)', hue='species', legend='full', ax=ax1[0, 0], palette=colorplot).set_title('Sepal (Actual)')
sns.scatterplot(data=iris, x='sepallength(cm)', y='sepalwidth(cm)', hue='pred_species', legend='full', ax=ax1[0, 1], palette=colorplot).set_title('Sepal (Predicted)')

sns.scatterplot(data=iris, x='petallength(cm)', y='petalwidth(cm)', hue='species', legend='full', ax=ax1[1, 0], palette=colorplot).set_title('Petal (Actual)')
sns.scatterplot(data=iris, x='petallength(cm)', y='petalwidth(cm)', hue='pred_species', legend='full', ax=ax1[1, 1], palette=colorplot).set_title('Petal (Predicted)')

plt.show()