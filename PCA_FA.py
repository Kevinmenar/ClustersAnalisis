import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from factor_analyzer import FactorAnalyzer

X = pd.read_csv('OutTest.csv', header = 0)
data_frame = X

scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
print(X_scaled)

columns_counts = 40

pca = PCA(n_components=columns_counts, copy=True)
pca_x = pca.fit_transform(X_scaled)
PC_values = np.arange(pca.n_components_) + 1
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

fa = FactorAnalyzer(columns_counts, rotation=None)
fa.fit(X_scaled)

#GET EIGENVALUES
ev = fa.get_eigenvalues()

ev_FA_str = " ".join(str(x) for x in ev[1])
ev_PCA_str = " ".join(str(x) for x in explained_variance)

textfile = open("a_file.txt", "w")
textfile.write(ev_FA_str + "\n")
textfile.write(ev_PCA_str + "\n")
textfile.close()