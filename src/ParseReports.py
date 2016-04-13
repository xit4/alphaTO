import json
import os
import pandas as pd
import numpy as np
import time

# here I use pandas to manipulate the reports

directory = '../Example Reports/reports'

# we know we're gonna have numberOfRows rows of data
numberOfRows = len([item for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item))])

# columns we are interested in
columnNames = ('sha',  # 'domains', # uncomment when needed
               'certificate')
# create DataFrame
df = pd.DataFrame(index=range(0, numberOfRows), columns=columnNames)

# initialize index counter
i = 0
# for each (report.json) file in the directory
for filename in os.listdir(directory):
    pathname = os.path.join(directory, filename)
    if not os.path.isfile(pathname):
        continue
    # open it
    with open(pathname) as data_file:
        # load the json data inside of it
        data = json.load(data_file)
        # find the domains used by the app
        if data['cuckoo']:
            for domain in data['cuckoo']['network']['domains']:
                dom = str(domain['domain']).rpartition('.')[2]
                if dom not in df.columns:
                    # place in position 2 just to skip sha and certificate, making it easier to ignore them when needed
                    df.insert(2, dom, pd.Series(np.zeros(numberOfRows, dtype=np.int8), index=df.index))
                df.set_value(i, dom, 1)

        # save the domains as strings one after the other, uncomment as needed
        # dom = ''
        # if data['cuckoo']:
        #     for domain in data['cuckoo']['network']['domains']:
        #         dom += domain['domain']
        # # save the domains in the DataFrame
        # df.set_value(i, 'domains', dom)

        # save the certificate serial in the DataFrame str(int(
        df.set_value(i, 'certificate', int(data['androguard']['certificate']['serial'], 16))
        # save the sha
        df.set_value(i, 'sha', data['sha256'])

        # find the permissions requested by the app
        for permission in data['androguard']['permissions']:
            # remove anything but the permission name
            permission = ''.join(x for x in permission if x.isupper() or x == '_')
            if permission not in df.columns:
                df[permission] = pd.Series(np.zeros(numberOfRows, dtype=np.int8), index=df.index)
            df.set_value(i, permission, 1)

    # update the counter
    i += 1
    # print a status message (mostly to keep ssh connection on... hopefully)
    print('\r' + '{0:.2f}% reports parsed '.format(i/numberOfRows*100), end='', flush=True)

print('')
start = time.time()
removedColumns = 0
threshold = 300
# for each column check if the sum is lower than threshold, if so, delete said column from original data
for columnName in df.columns.values:
    if columnName in columnNames:
        continue
    rowsum = df[columnName].sum(axis=1)
    if rowsum == 0 or rowsum < threshold:
        df.drop(columnName, axis=1, inplace=True)
        removedColumns += 1
end = time.time()
print('cleaning up the features took {0:.2f}s'.format(end-start))

# print the results in a CSV file
df.to_csv('../CSV/parsed.csv', index=False)
