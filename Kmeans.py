import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from validclust import dunn
from sklearn.metrics import pairwise_distances

fa_x = pd.read_csv('fa_loadings.csv', header = 0)
pca_x = pd.read_csv('pca_loadings.csv', header = 0)

max_clusters_FA = 20
silhoutte_list_FA = []
wcss_FA = []
DI_list_FA = []

for j in range(2,max_clusters_FA):
    kmeans = KMeans(n_clusters=j, random_state=0)
    kmeans.fit(fa_x)
    labels = kmeans.labels_
    dist = pairwise_distances(fa_x)
    dunn_index = dunn(dist, labels)
    DI_list_FA.append(dunn_index)
    silhoutte = silhouette_score(fa_x, labels, metric='sqeuclidean')
    silhoutte_list_FA.append(silhoutte)
    wcss_FA.append(kmeans.inertia_)

max_clusters_PCA = 20
silhoutte_list_PCA = []
wcss_PCA = []
DI_list_PCA = []

for j in range(2,max_clusters_PCA):
    kmeans = KMeans(n_clusters=j, random_state=0)
    kmeans.fit(pca_x)
    labels = kmeans.labels_
    dist = pairwise_distances(pca_x)
    dunn_index = dunn(dist, labels)
    DI_list_PCA.append(dunn_index)
    silhoutte = silhouette_score(pca_x, labels, metric='sqeuclidean')
    silhoutte_list_PCA.append(silhoutte)
    wcss_PCA.append(kmeans.inertia_)

DI_list_FA_str = " ".join(str(x) for x in DI_list_FA)
DI_list_PCA_str = " ".join(str(x) for x in DI_list_PCA)
print(DI_list_FA_str)
print(DI_list_PCA_str)

textfile = open("a_file.txt", "w")
textfile.write(DI_list_FA_str + "\n")
textfile.write(DI_list_PCA_str + "\n")
textfile.close()