import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.type_check import real
import pandas as pd

# https://en.wikipedia.org/wiki/Elo_rating_system#Theory

def column_find(data):
    """
    Find location of column headers, allowing for changing location of columns

    Args:
        data: csv data

    Returns:
       col_loc: dictionary of column names and their numeric location
    """
    n = 0
    cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC']
    col_loc = {}
    for i in data[0]:
        for j in cols:
            if i == j:
                col_loc[j] = n 
        n += 1
    return(col_loc)

def expected_outcome(winner, loser):
    """
    Estimate outcome of two teams given their rating

    Args:
        winner (float): Rating for winning team
        loser (foat): Rating for losing team

    Returns:
        float: Returns expected value of winner or 0.5 if draw
    """
    expectedWin = (1/(1+10**((loser - winner)/400)))
    expectedLoss = (1/(1+10**((winner - loser)/400)))
    expectedDraw = expectedWin - expectedLoss

    return(expectedWin, expectedLoss, expectedDraw)


def rating_update(expected_outcome, winner, loser, winner_kval, loser_kval, ind_variables):
    """
    Updates team ratings after a match

    Args:
        expected_outcome (float): Expected match outcome based on ratings
        winner (string): The team who won
        loser (string): The team who lost
        KVAL (int): 
        ind_variables (dict): Dictionary with variable name as key and coefficient as value
        gd_mult (float): Goal Difference multiplier

    Returns:
        float: Returns both updated team ratings
    """

    winner_rating = (winner + (winner_kval)*(expected_outcome[0]) + 
    (0.2*ind_variables['Goal Difference']) -
    (0.0657*ind_variables['Foul Difference']) + 
    (0.0867*ind_variables['Shots Taken']) +
    (0.0638*ind_variables['Corners']))

    # Subtract variable factors
    # e.g. if goal diff is negative, away team won, so the second line will add to their rating
    # and take away from home teams rating
    loser_rating = (loser - (loser_kval)*(expected_outcome[1]) - 
    (0.2*ind_variables['Goal Difference']) -
    (0.0657*ind_variables['Foul Difference']) + 
    (0.0867*ind_variables['Shots Taken']) +
    (0.0638*ind_variables['Corners']))

    return(winner_rating, loser_rating)

def rating_update_draw(expected_outcome, home, away, home_k, away_k, ind_variables):
    """
    Updates teams ratings if their games resulted in a draw

    Args:
        expected_outcome (float): Expected match outcome based on ratings
        home {string): Home team name
        away (string): Away team name
        home_k (int): Home team kval
        away_k (int): Away team kval
        ind_variables (dict): Dictionary with variable name as key and coefficient as value 
    """
    home_rating = (home - (home_k * (expected_outcome[0] - expected_outcome[1])))
    away_rating = (away - (away_k * (expected_outcome[1] - expected_outcome[0])))
    return(home_rating, away_rating)

def run_games(data, teams, cols, KVAL, ind_variables):
    """
    Takes sample data to create elo model

    Args:
        data (csv): CSV file
        teams (dict): Dictionary with variable name as key and coefficient as value
        columns (dict): 
        KVAL (int): 
        ind_variables (dict): Dictionary with variable name as key and coefficient as value

    Returns:
        dict: Returns both updated team ratings
    """

    # Loop through each data point (game)
    # skip last 300 as they are used for testing purposes
    for i in data[1:len(data) - 300]:
        home = i[cols['HomeTeam']]
        away = i[cols['AwayTeam']]
        winner = i[cols['FTR']]


        team_kval = [0,0]
        n = 0
        # Determines the kval each team gets based on their elo range
        for j in (teams[home], teams[away]):
            if j < 2100:
                team_kval[n] = KVAL[0]
            elif j in range(2100, 2400):
                team_kval[n] = KVAL[1]
            elif j > 2400:
                team_kval[n] = KVAL[2]
            n += 1

        # Calculate the independent variable differences
        ind_variables['Corners'] = int(i[cols['HC']]) - int(i[cols['AC']])
        ind_variables['Shots Taken'] = int(i[cols['HST']]) - int(i[cols['AST']])
        ind_variables['Foul Difference'] = int(i[cols['HF']]) - int(i[cols['AF']])
        ind_variables['Goal Difference'] = int(i[cols['FTHG']]) - int(i[cols['FTAG']])

        # Find results if winner is the home team
        if winner == '1':
            winner_kval = team_kval[0]
            loser_kval = team_kval[1]
            results = rating_update(expected_outcome(teams[home], teams[away]), teams[home], teams[away], winner_kval, loser_kval, ind_variables)
            teams[home] = results[0]
            teams[away] = results[1]

        # Find results if winner is the away team
        if winner == '0':
            winner_kval = team_kval[1]
            loser_kval = team_kval[0]
            results = rating_update(expected_outcome(teams[away], teams[home]), teams[away], teams[home], winner_kval, loser_kval, ind_variables)
            teams[home] = results[1]
            teams[away] = results[0]

        # Find results if game was a draw
        if winner == '0.5':
            home_k = team_kval[1]
            away_k = team_kval[0]
            
            results = rating_update_draw(expected_outcome(teams[home], teams[away]), teams[home], teams[away], home_k, away_k, ind_variables)
            teams[home] = results[0]
            teams[away] = results[1]

    return(teams)


def outcome_pr(teams, home_team, away_team, gap, home_advantage):
    """
    Compares two teams

    Args:
        teams (dict): Dictionary with variable name as key and coefficient as value
        home_team (string): The home team 
        away_team (string): The away team
        gap (int): the upper and lower bound of elos for a game to be a draw

    Returns:
        string: The match result
    """
    # Check the elo of the teams and weight the winner with a range for draws
    if teams[home_team] + home_advantage > teams[away_team] + gap:
        return(home_team)
    
    elif teams[home_team] + home_advantage < teams[away_team] - gap:
        return(away_team)
    
    else:
        return('Draw')


def ML_prediction(data, teams, cols, KVAL, GAP, HOME_ADVANTAGE):
    """
    Uses our model in predicting soccer game outcomes

    Args:
        data (list): The data for soccer matches
        teams (dict): Dictionary with variable name as key and coefficient as value
        cols (dict):
        KVAL (int): 
        gd_mult (float): Goal Difference multiplier

    Returns:
        string: Number of correct and incorrect predictions in a string
        float: The accuracy of the model
        float: The multiplier used for goal differences
    """
    correct = 0
    wrong = 0
    real_outcome = ''
    ind_variables = {}

    home_but_away = 0
    away_but_home = 0
    draw_but_not = 0
    notdraw_but_draw = 0

    # Looks at the last 300 games in our data set
    for i in data[len(data) - 300:]:
        home = i[cols['HomeTeam']]
        away = i[cols['AwayTeam']]
        winner = i[cols['FTR']]

        if winner == '1': real_outcome = home
        elif winner == '0': real_outcome = away
        elif winner == '0.5': real_outcome = 'Draw'

        # Get our models predicted winner
        prediction = outcome_pr(teams, home, away, GAP, HOME_ADVANTAGE)

        # Keep track of # of correct and wrong
        if prediction == real_outcome: correct += 1
        else: 
            wrong += 1
            if prediction == home and real_outcome == away:
                home_but_away += 1
            if prediction == away and real_outcome == home:
                away_but_home += 1
            if prediction == 'Draw' and real_outcome in (home, away):
                draw_but_not += 1
            if prediction in (home, away) and real_outcome == 'Draw':
                notdraw_but_draw += 1

            
        # The rest of this section adds updates the ELO rating based on the current game

        team_kval = [0,0]
        n = 0
        for j in (teams[home], teams[away]):
            if j < 2100:
                team_kval[n] = KVAL[0]
            elif j in range(2100, 2400):
                team_kval[n] = KVAL[1]
            elif j > 2400:
                team_kval[n] = KVAL[2]
            n += 1

        ind_variables['Corners'] = int(i[cols['HC']]) - int(i[cols['AC']])
        ind_variables['Shots Taken'] = int(i[cols['HST']]) - int(i[cols['AST']])
        ind_variables['Foul Difference'] = int(i[cols['HF']]) - int(i[cols['AF']])
        ind_variables['Goal Difference'] = int(i[cols['FTHG']]) - int(i[cols['FTAG']])
        
        if winner == '1':
            winner_kval = team_kval[0]
            loser_kval = team_kval[1]
            results = rating_update(expected_outcome(teams[home], teams[away]), teams[home], teams[away], winner_kval, loser_kval, ind_variables)
            teams[home] = results[0]
            teams[away] = results[1]

        if winner == '0':
            winner_kval = team_kval[1]
            loser_kval = team_kval[0]
            results = rating_update(expected_outcome(teams[away], teams[home]), teams[away], teams[home], winner_kval, loser_kval, ind_variables)
            teams[home] = results[1]
            teams[away] = results[0]

        if winner == '0.5':
            home_k = team_kval[1]
            away_k = team_kval[0]
            
            results = rating_update_draw(expected_outcome(teams[home], teams[away]), teams[home], teams[away], home_k, away_k, ind_variables)
            teams[home] = results[0]
            teams[away] = results[1]
    
    print('Predicted Home but Away: ' + str(round(home_but_away/wrong*100, 4)) + '%')
    print('Predicted Away but Home: ' + str(round(away_but_home/wrong*100, 4)) + '%')
    print('Predicted Draw but Not: ' + str(round(draw_but_not/wrong*100, 4)) + '%')
    print('Predicted Not Draw but Draw: ' + str(round(notdraw_but_draw/wrong*100, 4)) + '%')
    res1 = "The model has predicted %s games correctly and %s games incorrectly out of %s total games" %(correct, wrong, correct+wrong)
    accuracy = np.round(correct/(correct+wrong), 4)
    return(res1, accuracy)

def graph_output(teams):
    data = []
    for i in teams.values():
        data.append(int(i))

    min_val = min(data)
    max_val = max(data)
    
    plt.hist(data, bins=range(min_val, max_val + 45, 45), edgecolor="k")
    plt.title("ELO Distribution")
    plt.xlabel("ELO")
    plt.ylabel("# of Teams")
    #plt.xticks(bins)
    plt.show()

def main():
    # Base ELO
    ELO = 1200
    # KVAL for elo ranges, below 2100, 2100-2400, above 2400
    KVAL = [32, 24, 16]
    # Elo range for predicting draws
    GAP = 0
    # Bonus elo for a home team
    HOME_ADVANTAGE = 200
    ind_variables = {}

    file_path = 'C:/Users/codym/OneDrive - University of Calgary/Desktop/Econ 411 - Computer Applications/soccer data/soccer_data2.csv'
    with open(file_path, 'r', newline = '') as csv_f:
        data = list(csv.reader(csv_f))

    cols = column_find(data)

    # Get all teams in the csv file and give them a base elo rating of 1200
    teams = {}
    for i in data[1:]:
        if i[cols['HomeTeam']] not in teams:
            teams[i[cols['HomeTeam']]] = ELO
        if i[cols['AwayTeam']] not in teams:
            teams[i[cols['AwayTeam']]] = ELO

    print(run_games(data, teams, cols, KVAL, ind_variables))
    print(ML_prediction(data, teams, cols, KVAL, GAP, HOME_ADVANTAGE))

    graph_output(teams)

    with open('C:/Users/codym/OneDrive - University of Calgary/Desktop/Econ 411 - Computer Applications/soccer data/results.csv', 'w', newline = '') as csv_file:
        fieldnames = ['Team', 'ELO']
        writer = csv.DictWriter(csv_file, fieldnames = fieldnames)

        writer.writeheader()
        while len(teams) > 0:
            maxkey = max(teams, key=teams.get)
            writer.writerow({'Team':maxkey, 'ELO': str(round(teams[maxkey], 2))})
            print(maxkey + ": " + str(round(teams[maxkey], 2)))
            del teams[maxkey]
main()
