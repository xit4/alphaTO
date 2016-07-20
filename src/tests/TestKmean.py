import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pylab as pl

# here I test the Kmeans unsupervised model to compute the clusters

# extract data from the CSV, skip the first row cause it does not contain actual data and skip the last skip_footer
# lines to reduce the number of rows the computation has to process
csv = np.genfromtxt('../CSV/test3_permissions.csv', delimiter=',', skiprows=1, skip_footer=90000)

# remove the first column containing the SHAs which cannot be handled in a numpy array
data = csv[:, 1:]

# print(data, '\n', data.shape)

# reduce the data to two dimensions so that it can be easily represented in a 2D plot
pca = PCA(n_components=2)
pca.fit(data)
reduced = pca.transform(data)

# initialize the model asking for n_clusters
k_means = KMeans(n_clusters=5, random_state=0)
# fit the model to the data
k_means.fit(data)
# compute the clusters
predicted = k_means.predict(data)

# plot the results
print(predicted, predicted.shape)
pl.scatter(reduced[:, 0], reduced[:, 1], c=predicted, s=75, marker='s')
#pl.savefig('plot.jpg')  # save the plot as jpg
pl.show()

