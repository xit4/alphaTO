import json
import os
import pandas as pd
import io

# here I use pandas to manipulate the reports

stuff = 'domains,certificate\n'

for filename in os.listdir('../Example Reports'):
    pathname = '../Example Reports/' + filename
    with open(pathname) as data_file:
        data = json.load(data_file)
        domains = ''
        for domain in data['cuckoo']['network']['domains']:
            domains += domain['domain']
        if not domains: 
            domains = 'NaN'

        stuff += domains + ','
        stuff += data['androguard']['certificate']['serial'] + '\n'


# new_columns = pd.read_csv(io.StringIO(stuff), sep=',')
new_columns = pd.read_csv('../CSV/niggga.csv', sep=',')
# extract data from the CSV
csv = pd.read_csv('../CSV/test3_permissionsmodified.csv', sep=',', skipfooter=0)

# drop the first column (sha)
csv.drop(csv.columns[0], axis=1, inplace=True)

csv['domains'] = new_columns['domains']
csv['certificate'] = new_columns['certificate']

csv.to_csv('../CSV/OMG.csv', index=False)
#
# count = 0
# # for each column check if the sum is equal to 0, if so, delete said column from original data
# for name in csv.columns.values:
#     rowsum = csv[name].sum(axis=1)
#     if rowsum == 0 or rowsum < 70000:
#         csv.drop(name, axis=1, inplace=True)
#         count += 1
# print(count)
#
# # write resulting csv to file
# csv.to_csv('../CSV/test3_permissionsmodified.csv', index=False)
