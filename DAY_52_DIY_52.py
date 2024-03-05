#DIY DAY_52...
#Importing required libraries
import matplotlib.pyplot as plt  
import pandas as pd 
import seaborn as sns 
import numpy as np  
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

#Loading the CSV data into a DataFrame.

df = pd.read_csv('C:\\Users\\shiva\\Desktop\\Edurekha_data\\DAY_52\\DIY_DATASET\\Country-data.csv')
print(df.head())

X = df.drop('country',axis =1)  
X.head()

# Standardization of the dataset before performing PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:5,:5])


X_scaled_df = pd.DataFrame(X_scaled,columns=X.columns)
print(X_scaled_df)

import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))  
plt.title("Country Dendograms")  
dend = shc.dendrogram(shc.linkage(X_scaled_df, method='ward'))
plt.show()


# complete linkage
cl_mergings = linkage(X_scaled_df, method="complete", metric='euclidean')
plt.title("Complete Linkage Dendograms") 
dendrogram(cl_mergings)
plt.show()

# the single linkage clustering does not perform well in generating the clusters hence we go for complete linkage
# 4 clusters using complete linkage
cl_cluster_labels = cut_tree(cl_mergings, n_clusters=4).reshape(-1, )
cl_cluster_labels

X_scaled_df["Hierarchical_Cluster_labels"] = cl_cluster_labels
print(X_scaled_df)

#Performing PCA with 4 components

from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components=4)
X_pca_final = pca_final.fit_transform(X_scaled)
print(X.shape)
print(X_pca_final.shape)

#Adding cluster labels to the final PCA DataFrame.

X_pca_final_df = pd.DataFrame(X_pca_final,columns=['PC1','PC2','PC3','PC4'])
print(X_pca_final_df.head())


X_pca_final_df['Hierarchical_Cluster_Labels'] = cl_cluster_labels

print(X_pca_final_df.head())

#Analyzing how low GDP rate corresponds to the child mortality rate around the world.

ax = sns.scatterplot(x='gdpp',y='child_mort',data = X_scaled_df,hue='Hierarchical_Cluster_labels')
ax.set_xlabel('GDP', fontsize = 10)
ax.set_ylabel('Child Mortality Rate', fontsize = 10)
ax.set_title('How Low GDP Rate Corresponds to the Child Mortality Rate', fontsize = 15)
plt.show()
























