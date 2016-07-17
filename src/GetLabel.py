import json
import os
import pandas as pd
import numpy as np
import time
import argparse

# Script to extract labels from additional json files (details, detections, etc.)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputdir', required=True, help='directory containing json files (/ at the end)')
parser.add_argument('-o', '--outputfile', default='./parsedlabels.csv', help='the output file the results will be stored in')

inputdir = ''

args = parser.parse_args()

inputdir = args.inputdir
outputfile = args.outputfile

numberOfRows = len([item for item in os.listdir(inputdir)
                    if (os.path.isfile(os.path.join(inputdir, item)) and item.endswith('.json'))])

# columns we are interested in
columnNames = ('sha', 'label', 'labelname')
# create DataFrame
df = pd.DataFrame(index=range(0, numberOfRows), columns=columnNames)


start = time.time()
# initialize index counter
i = idx = 0

# for each (report.json) file in the directory
for filename in (os.listdir(inputdir)):
    pathname = os.path.join(inputdir, filename)
    if not os.path.isfile(pathname) or not filename.endswith('.json'):
        continue
    # open it
    with open(pathname) as data_file:
        # load the json data inside of it
        data = json.load(data_file)
        label = 0
        labelname = ""
        for av in data:
            for rule in av['rulesets']:
                label += rule['id']
                labelname += rule['name']
        if label:
            df.set_value(idx, 'sha', filename.replace("_detections", ""))
            df.set_value(idx, 'label', label)
            df.set_value(idx, 'labelname', labelname)
            idx += 1
    # update the counter
    i += 1
    # print a status message (mostly to keep ssh connection on... hopefully)
    print('\r' + '{0:.2f}% reports parsed '.format(i/numberOfRows*100), end='', flush=True)
print('')
end = time.time()
print('parsing {1} json took {0:.2f}s'.format(end-start, numberOfRows))

# print the results in a CSV file
df.to_csv(outputfile, index=False)