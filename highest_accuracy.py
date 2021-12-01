import csv
import numpy as np
import matplotlib.pyplot as plt

# https://en.wikipedia.org/wiki/Elo_rating_system#Theory


def expected_outcome(ratingA, ratingB):
    """
    Estimate outcome of two teams given their rating

    Args:
        ratingA (float): Rating for team A
        ratingB (foat): Rating for team B

    Returns:
        float: Returns expected value of winner or 0.5 if draw
    """
    expectedA = (1/(1+10**((ratingB - ratingA)/400)))
    expectedB = (1/(1+10**((ratingA - ratingB)/400)))

    if expectedA > expectedB:
        return expectedA
    
    if expectedA < expectedB:
        return expectedB
    
    if expectedA == expectedB:
        return 0.5


def rating_update(expected_outcome, actual_outcome, winner, loser, KVAL, ind_variables, gd_mult):
    """
    Updates team ratings after a match

    Args:
        expected_outcome (float): Expected match outcome based on ratings
        actual_outcome (int): Actual match outcome
        winner (string): The team who won
        loser (string): The team who lost
        KVAL (int): 
        ind_variables (dict): Dictionary with variable name as key and coefficient as value
        gd_mult (float): Goal Difference multiplier

    Returns:
        float: Returns both updated team ratings
    """
    
    winner_rating = (winner - (gd_mult*KVAL)*(actual_outcome - expected_outcome) + 
    ((gd_mult*KVAL) * abs(ind_variables['Goal Difference'])) -
    ((0.0657*KVAL) * ind_variables['Foul Difference']) + 
    ((0.0867*KVAL) * ind_variables['Shots Taken']) +
    ((0.0638*KVAL) * ind_variables['Corners']))

    # Subtract variable factors
    # e.g. if goal diff is negative, away team won, so the second line will add to their rating
    # and take away from home teams rating
    loser_rating = (loser - (gd_mult*KVAL)*((1-actual_outcome) - abs(expected_outcome - 1)) - 
    ((gd_mult*KVAL) * ind_variables['Goal Difference']) + 
    ((0.0657*KVAL) * ind_variables['Foul Difference']) + 
    ((0.0867*KVAL) * ind_variables['Shots Taken']) -
    ((0.0638*KVAL) * ind_variables['Corners']))

    return(winner_rating, loser_rating)


def run_games(data, teams, KVAL, ind_variables):
    """
    Takes sample data to create elo model

    Args:
        data (csv): CSV file
        teams (dict): Dictionary with variable name as key and coefficient as value
        KVAL (int): 
        ind_variables (dict): Dictionary with variable name as key and coefficient as value

    Returns:
        dict: Returns both updated team ratings
    """
    # Loop through each data point (game)
    # skip last 300 as they are used for testing purposes
    for i in data[1:len(data) - 300]:
        home = i[2]
        away = i[3]
        winner = i[6]

        ind_variables['Corners'] = int(i[17]) - int(i[18])
        ind_variables['Shots Taken'] = int(i[13]) - int(i[14])
        ind_variables['Foul Difference'] = int(i[15]) - int(i[16])
        ind_variables['Goal Difference'] = int(i[4]) - int(i[5])

        if winner == '1':
            results = rating_update(expected_outcome(teams[home], teams[away]), 1, teams[home], teams[away], KVAL, ind_variables)
            teams[home] = results[0]
            teams[away] = results[1]

        if winner == '0':
            results = rating_update(expected_outcome(teams[away], teams[home]), 0, teams[away], teams[home], KVAL, ind_variables)
            teams[home] = results[1]
            teams[away] = results[0]

    return(teams)


def outcome_pr(teams, home_team, away_team):
    """
    Compares two teams

    Args:
        teams (dict): Dictionary with variable name as key and coefficient as value
        home_team (string): The home team 
        away_team (string): The away team

    Returns:
        string: The match result
    """
    # Check the elo of the teams and weight the winner with a range for draws
    if teams[home_team] > teams[away_team]:
        return(home_team)
    
    elif teams[home_team] < teams[away_team]:
        return(away_team)
    else:
        return('Draw')


def ML_prediction(data, teams, KVAL, gd_mult):
    """
    Uses our model in predicting soccer game outcomes

    Args:
        data (list): The data for soccer matches
        teams (dict): Dictionary with variable name as key and coefficient as value
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

    # Looks at the last 300 games in our data set
    for i in data[len(data) - 300:]:
        home = i[2]
        away = i[3]
        winner = i[6]

        if winner == '1': real_outcome = home
        elif winner == '0': real_outcome = away
        elif winner == '0.5': real_outcome = 'Draw'

        # Get our models predicted winner
        prediction = outcome_pr(teams, home, away)

        # Keep track of # of correct and wrong
        if prediction == real_outcome: correct += 1
        else: wrong += 1
            
        # The rest of this section adds updates the ELO rating based on the current game

        ind_variables['Corners'] = int(i[17]) - int(i[18])
        ind_variables['Shots Taken'] = int(i[13]) - int(i[14])
        ind_variables['Foul Difference'] = int(i[15]) - int(i[16])
        ind_variables['Goal Difference'] = int(i[4]) - int(i[5])
        
        if winner == '1':
            results = rating_update(expected_outcome(teams[home], teams[away]), 1, teams[home], teams[away], KVAL, ind_variables, gd_mult)
            teams[home] = results[0]
            teams[away] = results[1]

        if winner == '0':
            results = rating_update(expected_outcome(teams[away], teams[home]), 0, teams[away], teams[home], KVAL, ind_variables, gd_mult)
            teams[home] = results[1]
            teams[away] = results[0]
    
    res1 = "The model has predicted %s games correctly and %s games incorrectly out of %s total games" %(correct, wrong, correct+wrong)
    accuracy = np.round(correct/(correct+wrong), 4)
    return(res1, accuracy, gd_mult)


def optimize(param, accuracy):
    """
    Finds the optimal multiplier for a parameter

    Args:
        param (list): The values of the parameters
        accuracy (list): The accuracy scores that correspond to the parameters

    Returns:
        float: The optimal parameter for maximum accuracy
    """
    param_array = np.array(param)
    accuracy_array = np.array(accuracy)

    # Finds index with highest accuracy
    optimal_index = np.argmax(accuracy_array)
    return param_array[optimal_index]


def plot_optimal(param, accuracy):
    """
    Plots a line graph for different values of a parameter and model accuracy

    Args:
        param (list): The values of the parameters
        accuracy (list): The accuracy scores that correspond to the parameters
    """
    plt.figure(figsize=(10,6))
    plt.plot(param, accuracy)
    
    plt.title('Goal Difference Multiplier Affect on Accuracy', 
              fontsize=12)
    plt.xlabel('Multiplier', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    
    plt.savefig('GD_Acc.jpg')
    plt.show()


def main():
    KVAL = 15
    ind_variables = {}

    file_path = 'github_clone/soccer_data2.csv'
    with open(file_path, 'r', newline = '') as csv_f:
        data = list(csv.reader(csv_f))

    # Get all teams in the csv file and give them a base elo rating of 1200
    teams = {}
    for i in data[1:]:
        if i[2] not in teams:
            teams[i[2]] = 1200
    
    # print(run_games(data, teams, KVAL, ind_variables))
    
    gd_optimize_list = []
    accuracy_list = []
    gd_vals = np.arange(0, 1, 0.002)  # (start, end, step)
    
    # Find optimal goal difference estimate
    for gd_mult in gd_vals:
        res1, accuracy, gd_mult = ML_prediction(data, teams, KVAL, gd_mult)
        gd_optimize_list.append(gd_mult)
        accuracy_list.append(accuracy)
    
    gd_optimize_array = np.array(gd_optimize_list)
    accuracy_array = np.array(accuracy_list)
    
    plot_optimal(gd_optimize_array, accuracy_array)
    
    # Hyperparameter tuning
    gd_optimal = optimize(gd_optimize_array, accuracy_array)
    
    # Get optimized results
    res1, accuracy, gd_mult = ML_prediction(data, teams, KVAL, gd_optimal)
    accuracy = np.round(accuracy*100, 4)
    
    print(res1)
    print('Model accuracy: %s percent' %(accuracy))


main()
