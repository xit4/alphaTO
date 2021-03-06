import json
import os
import pandas as pd
import numpy as np
import time
import argparse

# here I use pandas to parse the reports and print features in a CSV
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputdir', required=True, help='directory containing json files (/ at the end)')
parser.add_argument('-f', '--filter', help='text file containing the names of the json files to be parsed (one file per line')
parser.add_argument('-o', '--outputfile', help='the output directory the results will be stored in (/ at the end)')

inputdir = ''
outputfile = './parsed.csv'
filterfile = ''
filesList = list()

args = parser.parse_args()

inputdir = args.inputdir
if args.outputfile:
    outputfile = args.outputfile
if args.filter:
    filterfile = args.filter

# find out how many samples we have
numberOfRows = 0
# use filter file if it is provided
if filterfile:
    filesList = list()
    with open(filterfile, 'r') as f:
        for line in f:
            numberOfRows += 1
            filesList.append(line[:-1])
# otherwise count the json file in the input directory
else:
    numberOfRows = len([item for item in os.listdir(inputdir)
                        if (os.path.isfile(os.path.join(inputdir, item)) and item.endswith('.json'))])

# columns we are interested in
columnNames = ('sha', 'name', 'package', 'certificate')
# create DataFrame
df = pd.DataFrame(index=range(0, numberOfRows), columns=columnNames)

# initialize index counter
i = 0
# for each (report.json) file in the directory
for filename in (filesList if filesList else os.listdir(inputdir)):
    pathname = os.path.join(inputdir, filename)
    if not os.path.isfile(pathname) or not filename.endswith('.json'):
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
                    # place in position 3 just to skip sha and certificate, making it easier to ignore them when needed
                    df.insert(len(columnNames), dom, pd.Series(np.zeros(numberOfRows, dtype=np.int8), index=df.index))
                df.set_value(i, dom, 1)

        # save the domains as strings one after the other, uncomment as needed
        # dom = ''
        # if data['cuckoo']:
        #     for domain in data['cuckoo']['network']['domains']:
        #         dom += domain['domain']
        # # save the domains in the DataFrame
        # df.set_value(i, 'domains', dom)

        # save the sha
        df.set_value(i, 'sha', data['sha256'])
        df.set_value(i, 'name', data['androguard']['app_name'])
        df.set_value(i, 'package', data['androguard']['package_name'])
        df.set_value(i, 'certificate', data['androguard']['certificate']['serial'])

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
print('removed {0} columns'.format(removedColumns))

# print the results in a CSV file
df.to_csv(outputfile, index=False)
