
import csv

file_path = 'C:/Users/codym/OneDrive - University of Calgary/Desktop/Econ 411 - Computer Applications/soccer data/soccer_data.csv'

with open(file_path, 'r', newline = '') as csv_f:
    data = list(csv.reader(csv_f))

# print headers 
print(data[0])
