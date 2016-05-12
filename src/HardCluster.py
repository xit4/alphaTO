from sklearn import cluster
import time
import pandas as pd
import numpy as np


# read the rows from the CSV file. Skip some in skipfooter to reduce computational times/memory requirements
df = pd.read_csv('../parsedstats.csv', sep=',', header=0, engine='python',  skipfooter=0)

# skip sha, name, certificate and package
data = df[df.columns[4:]].values

eps_range = np.linspace(0.1, 10, 40)
min_sample_range = range(7)

for eps in eps_range:
    for min_sample in min_sample_range:
        model = cluster.DBSCAN(eps=eps, min_samples=min_sample)
        start = time.time()
        # fit the date and compute compute the clusters
        predicted = model.fit_predict(data)
        end = time.time()
        print('DBSCAN execution time ', end-start)
        df.insert(1, 'cluster', predicted)
        # df.to_csv('../CSV/DBSCANeps{0}.csv'.format(eps), index=False)

        count = 1
        for row in df['cluster']:
            if row == -1:
                count += 1
        df.drop('cluster', axis=1, inplace=True)

        print('Number of outliers ', count, ' eps=', eps, ' min_samples=', min_sample)
        print('Total number of clusters ', max(predicted))
