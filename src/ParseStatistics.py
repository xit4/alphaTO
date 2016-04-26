import json
import os
import pandas as pd
import numpy as np
import time
import argparse

# here I use pandas to parse the reports and print statistics of them in a CSV
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputdir', required=True, help='directory containing json files (/ at the end)')
parser.add_argument('-f', '--filter', help='text file containing the names of the json files to be parsed (one file per line')
parser.add_argument('-o', '--outputdir', help='the output directory the results will be stored in (/ at the end)')

inputdir = ''
outputdir = './'
filterfile = ''
filesList = list()

args = parser.parse_args()

inputdir = args.inputdir
if args.outputdir:
    outputdir = args.outputdir
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
columnNames = ('sha', 'name', 'package', 'certificate', 'filters', 'activities', 'receivers', 'services', 'permissions',
               'http', 'hosts', 'domains', 'dns', 'fileswritten', 'cryptousage', 'filesread', 'sendsms', 'sendnet',
               'recvnet')
# create DataFrame
df = pd.DataFrame(index=range(0, numberOfRows), columns=columnNames)


start = time.time()
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

        if data['androguard']:
            df.set_value(i, 'sha', data['sha256'])
            df.set_value(i, 'name', data['androguard']['app_name'])
            df.set_value(i, 'package', data['androguard']['package_name'])
            df.set_value(i, 'certificate', data['androguard']['certificate']['serial'])
            df.set_value(i, 'filters', len(data['androguard']['filters']))
            df.set_value(i, 'activities', len(data['androguard']['activities']))
            df.set_value(i, 'receivers', len(data['androguard']['receivers']))
            df.set_value(i, 'services', len(data['androguard']['services']))
            df.set_value(i, 'permissions', len(data['androguard']['permissions']))

        if data['cuckoo']:
            df.set_value(i, 'http', len(data['cuckoo']['network']['http']))
            df.set_value(i, 'hosts', len(data['cuckoo']['network']['hosts']))
            df.set_value(i, 'domains', len(data['cuckoo']['network']['domains']))
            df.set_value(i, 'dns', len(data['cuckoo']['network']['dns']))


        if data['droidbox']:
            df.set_value(i, 'fileswritten', len(data['droidbox']['fileswritten']))
            df.set_value(i, 'cryptousage', len(data['droidbox']['cryptousage']))
            df.set_value(i, 'fileswritten', len(data['droidbox']['fileswritten']))
            df.set_value(i, 'filesread', len(data['droidbox']['filesread']))
            df.set_value(i, 'sendsms', len(data['droidbox']['sendsms']))
            df.set_value(i, 'sendnet', len(data['droidbox']['sendnet']))
            df.set_value(i, 'recvnet', len(data['droidbox']['recvnet']))

    # update the counter
    i += 1
    # print a status message (mostly to keep ssh connection on... hopefully)
    print('\r' + '{0:.2f}% reports parsed '.format(i/numberOfRows*100), end='', flush=True)
print('')
end = time.time()
print('parsing {1} json took {0:.2f}s'.format(end-start, numberOfRows))

start = time.time()
removedColumns = 0
threshold = 0
# for each column check if the sum is lower than threshold, if so, delete said column from original data
for columnName in df.columns.values:
    if columnName in columnNames:
        continue
    rowsum = df[columnName].sum(axis=1)
    if rowsum == 0 or rowsum < threshold:
        df.drop(columnName, axis=1, inplace=True)
        removedColumns += 1
end = time.time()
df.fillna(value=0, inplace=True)
print('cleaning up the features took {0:.2f}s'.format(end-start))
print('removed {0} columns'.format(removedColumns))

# print the results in a CSV file
df.to_csv(outputdir+'parsedstats.csv', index=False)
