import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def preproccessing(dataframe):
    '''
    Takes messy data and cleans it

    Args:
        df (csv): CSV file
    '''
    # Drop all duplicated rows
    soccer_no_dup = soccer.drop_duplicates().iloc[:-1]

    # Create dummy variables
    dummy_home = pd.get_dummies(soccer_no_dup.HomeTeam, drop_first=True, prefix='HomeTeam')
    dummy_away = pd.get_dummies(soccer_no_dup.AwayTeam, drop_first=True, prefix='AwayTeam')
    dummy_results = pd.get_dummies(soccer_no_dup.FTR, drop_first=True, prefix='FTR')
    dummy_halftime_res = pd.get_dummies(soccer_no_dup.HTR, drop_first=True, prefix='HTR')
    dummy_referee = pd.get_dummies(soccer_no_dup.Referee, drop_first=True, prefix='Referee')

    # Merge dummy data frames with original soccer data frame
    # \ lets you continue code on a new line
    soccer_no_dup = pd.merge(soccer_no_dup, dummy_home, left_index=True, right_index=True) \
        .merge(dummy_away, left_index=True, right_index=True) \
        .merge(dummy_results, left_index=True, right_index=True) \
        .merge(dummy_halftime_res, left_index=True, right_index=True) \
        .merge(dummy_referee, left_index=True, right_index=True)

    # Drop columns used for creating dummy variables
    to_drop = ['HomeTeam', 'AwayTeam', 'FTR', 'HTR', 'Referee']
    soccer_no_dup.drop(columns=to_drop, inplace=True)

    # Feature engineer new variables
    # Note: series.sub(other series) = series - other series
    soccer_no_dup['Halftime Goal Difference'] = soccer_no_dup.HTHG.sub(soccer_no_dup.HTAG)
    soccer_no_dup['Shot Difference'] = soccer_no_dup.HS.sub(soccer_no_dup.AS)
    soccer_no_dup['Shot On Target Difference'] = soccer_no_dup.HST.sub(soccer_no_dup.AST)
    soccer_no_dup['Foul Difference'] = soccer_no_dup.HF.sub(soccer_no_dup.AF)
    soccer_no_dup['Corner Difference'] = soccer_no_dup.HC.sub(soccer_no_dup.AC)
    soccer_no_dup['Yellow Card Difference'] = soccer_no_dup.HY.sub(soccer_no_dup.AY)

    # Drop variables used for feature engineering
    to_drop = ['HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY']
    soccer_no_dup.drop(columns=to_drop, inplace=True)   


def run_model():
    '''
    Runs a random forest model and returns the estimated coefficients of feature importance
    
    Returns:
        Series (pd.Series): Pandas series object
    '''
    # Independent and dependent variables
    X = soccer_no_dup.drop(columns=['Div', 'Date', 'FTHG', 'FTAG','FTR_D', 'FTR_H'])
    y = soccer_no_dup[['FTR_D', 'FTR_H']]

    # Splits the data into 70% training and 30% testing
    # Note: random_state lets you and others using this code to generate the same results
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    # Fit model to data
    rf = RandomForestClassifier(random_state=123)
    rf.fit(X_train, y_train)

    # Returns most importance features sorted
    return pd.Series(rf.feature_importances_, index=X_train.columns).sort_values()


def plot_feat_imp(series):
    '''
    Plots the feature importance coefficients
    
    Args:
        series (pd.Series): Pandas series object
    '''
    # Plotting results
    plt.figure(figsize=(16,6))
    
    series.iloc[-10:].plot(kind='barh')  # Showing only top ten
    plt.title('Feature Importance', fontsize=14)
    plt.xlabel('Coefficient', fontsize=12)

    # Place text values in plot
    for i, val in enumerate(series.iloc[-10:]):
        plt.text(x=val-0.005, y=i-0.1,
                s=f'{np.round(val, 4)}',
                color='white')

    # Saves plot into working directory
    # plt.savefig('feat_import.jpg')
    plt.show()


def main():    
    # Read in data
    soccer = pd.read_csv(os.getcwd() + '/data/soccer_data.csv', parse_dates=['Date'])
    # Clean data
    preproccessing(soccer)
    # Get feature importances
    series = run_model()
    # Plot results
    plot_feat_imp(series)


main()
