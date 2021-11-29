
import csv
import numpy as np

# Function to estimate outcome of two teams given their rating
# Takes two ratings as parameters
# Returns the expected outcome of their game, winning team and expected probability of winning
# https://en.wikipedia.org/wiki/Elo_rating_system#Theory
def expected_outcome(ratingA, ratingB):
    outcome = ''
    expectedA = (1/(1+10**((ratingB - ratingA)/400)))
    expectedB = (1/(1+10**((ratingA - ratingB)/400)))

    if expectedA > expectedB:
        return(outcome, expectedA)
    
    if expectedA < expectedB:
        return(outcome, expectedB)
    
    if expectedA == expectedB:
        return(outcome, 0.5)
    
# Function to update two teams ratings after a match
def rating_update(expected_outcome, actual_outcome, winner, loser, KVAL, ind_variables):
    
    winner_rating = (winner + (0.6511*KVAL)*(actual_outcome - expected_outcome) + 
    ((0.2*KVAL) * ind_variables['Goal Difference'])+ 
    ((0.0453*KVAL) * ind_variables['Fouls']) + 
    ((0.0611*KVAL) * ind_variables['Shots Taken']) +
    ((0.0425*KVAL) * ind_variables['Corners']))


    loser_rating = (loser + (0.6511*KVAL)*((1-actual_outcome) - abs(expected_outcome - 1)) + 
    ((0.2*KVAL) * ind_variables['Goal Difference']) + 
    ((0.0453*KVAL) * ind_variables['Fouls']) + 
    ((0.0611*KVAL) * ind_variables['Shots Taken']) +
    ((0.0425*KVAL) * ind_variables['Corners']))

    return(winner_rating, loser_rating)

# Function that takes sample data to create our elo model
# Takes csv data, teams dictionary, kvalue, and all independent variables as parameters
# Returns the dictionary of teams and their elos with our sample data
def run_games(data, teams, KVAL, ind_variables):

    # Loop through each data point (game)
    # skip last 300 as they are used for testing purposes
    for i in data[1:len(data) - 200]:
        home = i[2]
        away = i[3]
        winner = i[6]

        goals_home = int(i[4])
        goals_away = int(i[5])

        if winner == '1':
            ind_variables['Corners'] = int(i[17]) - int(i[18])
            ind_variables['Shots Taken'] = int(i[13]) - int(i[14])
            ind_variables['Fouls'] = int(i[15]) - int(i[16])
            ind_variables['Goal Difference'] = int(goals_home) - int(goals_away)
            results = rating_update(expected_outcome(teams[home], teams[away])[1], 1, teams[home], teams[away], KVAL, ind_variables)
            teams[home] = results[0]
            teams[away] = results[1]

        if winner == '0':
            ind_variables['Corners'] = int(i[18]) - int(i[17])
            ind_variables['Shots Taken'] = int(i[14]) - int(i[13])
            ind_variables['Fouls'] = int(i[16]) - int(i[15]) 
            ind_variables['Goal Difference'] = int(goals_away) - int(goals_home)
            results = rating_update(expected_outcome(teams[away], teams[home])[1], 0, teams[away], teams[home], KVAL, ind_variables)
            teams[home] = results[1]
            teams[away] = results[0]


    return(teams)

# function to compare 2 teams
def outcome_pr(teams, home_team, away_team):

    # Check the elo of the teams and weight the winner with a range for draws
    if teams[home_team] > teams[away_team] + 50:
        #print(f'{home_team} will win')
        return(home_team)
    
    elif teams[home_team] < teams[away_team] - 50:
        #print(f'{away_team} will win')
        return(away_team)
    
    else:
        return('Draw')

# Function to use our model in predicting soccer game outcomes
# Takes csv data and a dictionary of teams elo scores as parameters
def ML_prediction(data, teams, KVAL):
    correct = 0
    wrong = 0
    real_outcome = ''
    ind_variables = {}

    # Looks at the last 300 games in our data set
    for i in data[len(data) - 200:]:
        home = i[2]
        away = i[3]
        winner = i[6]

        if winner == '1': real_outcome = home
        elif winner == '0': real_outcome = away
        elif winner == '0.5': real_outcome = 'Draw'

        # Get our models predicted winner
        prediction = outcome_pr(teams, home, away)

        # Keep track of # of correct and wrong
        if prediction == real_outcome:
            correct += 1
        else:
            wrong += 1

        # The rest of this section adds updates the ELO rating based on the current game
        goals_home = int(i[4])
        goals_away = int(i[5])


        if winner == '1':
            ind_variables['Corners'] = int(i[17]) - int(i[18])
            ind_variables['Shots Taken'] = int(i[13]) - int(i[14])
            ind_variables['Fouls'] = int(i[15]) - int(i[16])
            ind_variables['Goal Difference'] = int(goals_home) - int(goals_away)
            results = rating_update(expected_outcome(teams[home], teams[away])[1], 1, teams[home], teams[away], KVAL, ind_variables)
            teams[home] = results[0]
            teams[away] = results[1]

        if winner == '0':
            ind_variables['Corners'] = int(i[18]) - int(i[17])
            ind_variables['Shots Taken'] = int(i[14]) - int(i[13])
            ind_variables['Fouls'] = int(i[16]) - int(i[15])
            results = rating_update(expected_outcome(teams[away], teams[home])[1], 0, teams[away], teams[home], KVAL, ind_variables)
            teams[home] = results[1]
            teams[away] = results[0]
        
    print("The model has predicted %s games correctly and %s games incorrectly" %(correct,wrong))
    print("The result is a %s" %(correct/(correct+wrong)))

def main():
    KVAL = 15
    ind_variables = {}

    file_path = 'C:/Users/codym/OneDrive - University of Calgary/Desktop/Econ 411 - Computer Applications/soccer data/soccer_data2.csv'
    with open(file_path, 'r', newline = '') as csv_f:
        data = list(csv.reader(csv_f))

    # Get all teams in the csv file and give them a base elo rating of 1200
    teams = {}
    for i in data[1:]:
        if i[2] not in teams:
            teams[i[2]] = 1200

    print(run_games(data, teams, KVAL, ind_variables))
    ML_prediction(data, teams, KVAL)

main()
