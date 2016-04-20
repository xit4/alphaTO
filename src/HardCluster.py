from sklearn import cluster
import time
import pandas as pd


# extract data from the CSV, skip the first row cause it does not contain actual data and skip the last 50000 lines to
# reduce the number of rows the computation has to process
df = pd.read_csv('./parsed.csv', sep=',', header=0, engine='python',  skipfooter=0)

# skip sha, name and package
data = df[df.columns[4:]].values

# ----------------------------
# AgglomerativeClustering (uncomment as needed)
dbscan = cluster.DBSCAN()
start = time.time()
# fit the date and compute compute the clusters
predicted = dbscan.fit_predict(data)
end = time.time()
print('DBSCAN execution time ', end-start)

df.insert(1, 'cluster', predicted)
df.to_csv('./clusterizeddbscan.csv', index=False)
df.drop('cluster', axis=1, inplace=True)

# ----------------------------
# AgglomerativeClustering (uncomment as needed)
agglomerativeclustering = cluster.AgglomerativeClustering(n_clusters=30)
start = time.time()
# fit the date and compute compute the clusters
predicted = agglomerativeclustering.fit_predict(data)
end = time.time()
print('AgglomerativeClustering with 30 clusters execution time ', end-start)

df.insert(1, 'cluster', predicted)
df.to_csv('./clusterizedAgglo30.csv', index=False)
df.drop('cluster', axis=1, inplace=True)

# ----------------------------
# AgglomerativeClustering (uncomment as needed)
agglomerativeclustering = cluster.AgglomerativeClustering(n_clusters=60)
start = time.time()
# fit the date and compute compute the clusters
predicted = agglomerativeclustering.fit_predict(data)
end = time.time()
print('AgglomerativeClustering with 60 clusters execution time ', end-start)

df.insert(1, 'cluster', predicted)
df.to_csv('./clusterizedAgglo60.csv', index=False)