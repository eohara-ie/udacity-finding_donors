import csv

dates = []
scores = []

with open('datareg.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        dates.append(row[0])
        scores.append(row[1])

print(dates)
print('\n')
print(scores)