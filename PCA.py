import pandas as pd 
import numpy as np
wine = pd.read_csv("D:\\CQ\\Python\\datasets\\wine.csv")
wine.describe()
wine.head()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
wine_data = wine.iloc[:,1:]
wine_data.head(4)

# Normalizing the numerical data 
wine_normal = scale(wine_data)

pca = PCA(n_components = 13)
pca_values = pca.fit_transform(wine_normal)

pca.explained_variance_ratio_
# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
np.cumsum(var)

# Cumulative variance 

var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1



# plot between PCA1 and PCA2 
x = pca_values[:,0]
y =pca_values[:,1]
z = pca_values[:,2]
plt.scatter(x,y)
np.corrcoef(x,y)



################### Clustering  ##########################
new_df = pd.DataFrame(pca_values[:,0:4])

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3)
kmeans.fit(new_df)
kmeans.labels_
