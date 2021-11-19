import csv
import numpy as np
from scipy.linalg import lstsq

sample=["G","F","M"] #create list of possible samples

again = "Y" #create while loop 
while again == "Y": #start while loop

    with open('C:/Users/marta/Downloads/cps_small.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))

    data_array = np.array(data) #create aray with all data
    varnames = data_array[0,:] #create array with the variable names
    data_values = data_array[1:, :].astype(float) #create array with data values
    varnames2 = varnames[1:] #create list of variable names excluding wages

    print ("What type of regression do you want to run? Please type 'G' for general, 'F' for female or 'M' for male") 
    gender=input().upper() #ask for type of regression

    while gender not in sample: #condition to just accept general, women or female regression 
        print("Please enter a valid sample") 
        gender=input().upper() 
   
    if sample[1] == gender: #condition to run a female regression

        data_values = data_values[data_values[:,5]==1] #get data from just women
        data_values = np.delete(data_values,5,1) #delete female column 
        varnames = np.delete(varnames,5) #delete female from varnames list
        print("WOMEN'S SAMPLE: ")
       
    elif sample[2] == gender:

        data_values = data_values[data_values[:,5]==0] #get data from just man
        data_values = np.delete(data_values,5,1) #delete female column
        varnames = np.delete(varnames,5) #delete female from varnames list
        print("MEN'S SAMPLE: ")

    #run general regression otherwise since you don't need to restrict your population
    
    independent= np.array([]) #create empty array

    i=1 #Since 0 would correspond to wage, we need to start at 1, which corresponds to the first independent variable
        
    for i in range (len(varnames)): #condition to run bivariate regressions on every independent variable

        xvalues = data_values[:, varnames == varnames[i]] #get every value for each of the independent variables 
        constant = np.ones((len(xvalues),1)) #generate a intercept array with value 1 and lenght equal to number of independent variables
        xvalues = np.hstack((constant, xvalues)) #horizontal stack of intercepts and independent variables
        
        yvalues = data_values[:, varnames=="wage"] #obtain dependent variable = wage

        betas, res, rnk, s = lstsq(xvalues, yvalues) #run least squares regression and obtain betas
    
        independent = np.append(independent, betas[1]) #add obtained result to empty array 

    trans_independent = independent.reshape(-1,1) #transpose independent array 
    trans_varnames = varnames.reshape(-1,1) #transpose variable names array
    trans_full = np.hstack((trans_varnames,trans_independent)) #horizontal stack of transposed variable names and betas 
    print(trans_full) #print horizontal stack of variable names and betas

    
    again= input("Do you want to run another regression? If yes, please type 'Y' ").upper() #ask if you want to run another regression



    


