import pandas as pd

# here I test pandas to manipulate the features
# in this particular file, I sum all the columns and remove those that result in a 0 sum

# extract data from the CSV
csv = pd.read_csv('../CSV/test3_permissions.csv', sep=',', skipfooter=0)

# drop the first column (sha)
csv.drop(csv.columns[0], axis=1, inplace=True)

count = 0
# for each column check if the sum is equal to 0, if so, delete said column from original data
for name in csv.columns.values:
    rowsum = csv[name].sum(axis=1)
    if rowsum == 0 or rowsum < 70000:
        csv.drop(name, axis=1, inplace=True)
        count += 1
print(count)

# write resulting csv to file
csv.to_csv('../CSV/test3_permissionsmodified.csv', index=False)
