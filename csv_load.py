
import csv

file_path = 'C:/Users/codym/OneDrive - University of Calgary/Desktop/Econ 411 - Computer Applications/soccer data/soccer_data.csv'

with open(file_path, 'r', newline = '') as csv_f:
    data = list(csv.reader(csv_f))

# Get all the team names and give them a base elo score of 1200    
teams = {}
for i in data[1:]:
    if i[2] not in teams:
        teams[i[2]] = 1200

for i in data[2:]:
    if i[2] not in teams:
        teams[i[2]] = 1200

print(teams)
