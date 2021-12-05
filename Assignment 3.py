 
  
import csv
import numpy as np
from scipy.linalg import lstsq

with open('C:/Users/marta/OneDrive/Documentos/Libro1.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile, delimiter = ';'))

data_array = np.array(data) #create aray with all data
varnames = data_array[0,:] #create array with the variable names
data_values = data_array[1:, :].astype(float) #create array with data values
varnames2 = varnames[1:] #create list of variable names excluding wages

    
    
independent= np.array([]) #create empty array

i=1 #Since 0 would correspond to wage, we need to start at 1, which corresponds to the first independent variable
        
for i in range (len(varnames)): #condition to run bivariate regressions on every independent variable
    xvalues = data_values[:, varnames == varnames[i]] #get every value for each of the independent variables 
    constant = np.ones((len(xvalues),1)) #generate a intercept array with value 1 and lenght equal to number of independent variables
    xvalues = np.hstack((constant, xvalues)) #horizontal stack of intercepts and independent variables
    
    yvalues = data_values[:, varnames=="FTR"] #obtain dependent variable = wage
    betas, res, rnk, s = lstsq(xvalues, yvalues) #run least squares regression and obtain betas
    
    independent = np.append(independent, betas[1]) #add obtained result to empty array 

trans_independent = independent.reshape(-1,1) #transpose independent array 
trans_varnames = varnames.reshape(-1,1) #transpose variable names array
trans_full = np.hstack((trans_varnames,trans_independent)) #horizontal stack of transposed variable names and betas 
print(trans_full) #print horizontal stack of variable names and betas

    
   

    






    


