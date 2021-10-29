# This function is used to change the value of the FTR column, if home team won the
# value is = 1, any other value means home team lost or had a draw

# Open the original CSV File first
file_path = 'C:/Users/codym/OneDrive - University of Calgary/Desktop/Econ 411 - Computer Applications/soccer data/soccer data.csv'
with open(file_path, 'r', newline = '') as csv_f:
    data = list(csv.reader(csv_f))

# Set the filepath of your new CSV file, make sure the file has a different name!
file_path = 'C:/Users/codym/OneDrive - University of Calgary/Desktop/Econ 411 - Computer Applications/soccer data/soccer_data2.csv'
with open(file_path, "w", newline = '') as csv_f:
    datawriter = csv.writer(csv_f, delimiter = ',')
    rowdata = []
    # Loop through each row
    for i in data:
        rowdata = []
        n = 0
        # Loop through each column of current row
        for j in i:
            # Check if using the FTR column
            if n == 6:
                # Set home team wins to 1, all else to 0
                if j == 'H':
                    j = 1
                else:
                    j = 0
            rowdata.append(j)
            n += 1
        datawriter.writerow(rowdata)
