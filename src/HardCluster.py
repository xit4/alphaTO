from sklearn import cluster
import time
import pandas as pd


# read the rows from the CSV file. Skip some in skipfooter to reduce computational times/memory requirements
df = pd.read_csv('../CSV/parsedstats.csv', sep=',', header=0, engine='python',  skipfooter=0)

# skip sha, name, certificate and package
data = df[df.columns[4:]].values

# ----------------------------
# DBSCAN (uncomment as needed)
dbscan = cluster.DBSCAN()
start = time.time()
# fit the date and compute compute the clusters
predicted = dbscan.fit_predict(data)
end = time.time()
print('DBSCAN execution time ', end-start)

df.insert(1, 'cluster', predicted)
df.to_csv('../CSV/clusterized.csv', index=False)