"""
Econ 411 - Soccer Game Elo Machine Learning Project
2021/12/12
Cody Moldenhauer, Saul Chirinos, Marta Centenera, Carolyn Zhao, Shea Murphy, Shaan Bajwa

The purpose of this project is to take sample data of soccer games and develop an
elo system used in our algorithm to predict the outcome of future games.
The basis of our elo system was created in similarity to the chess method:
https://en.wikipedia.org/wiki/Elo_rating_system#Theory
"""
import csv
import numpy as np
import matplotlib.pyplot as plt


def column_find(data):
    """
    Find location of column headers, allowing for changing location of columns

    Args:
        data: csv data
    Returns:
       col_loc: dictionary of column names and their numeric location
    """
    n = 0
    cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'Date']
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
        expectedWin (float): probability of team winning
        expectedLoss (float): probability of team losing
        expectedDraw (float): probability of a draw
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
        winn_kval (int): winning teams kvalue
        loser_kval (int): losing teams kvalue
        ind_variables (dict): Dictionary with variable name as key and coefficient as value
    Returns:
        winner_rating (int): winning teams new elo
        loser_rating (int): losing teams new elo
    """

    winner_rating = (winner + (winner_kval)* (expected_outcome[0]) + 
    (0.2*ind_variables['Goal Difference']) -
    (0.004*ind_variables['Foul Difference']) + 
    (0.013*ind_variables['Shots Taken']) +
    (0.005*ind_variables['Corners']))

    # Subtract variable factors
    # e.g. if goal diff is negative, away team won, so the second line will add to their rating
    # and take away from home teams rating
    loser_rating = (loser - (loser_kval)*(expected_outcome[1]) - 
    (0.2*ind_variables['Goal Difference']) -
    (0.004*ind_variables['Foul Difference']) - 
    (0.013*ind_variables['Shots Taken']) *
    (0.005*ind_variables['Corners']))

    return(winner_rating, loser_rating)


def rating_update_draw(expected_outcome, home, away, home_k, away_k, ind_variables):
    """
    Updates teams ratings if their games resulted in a draw

    Args:
        expected_outcome (float): Expected match outcome based on ratings
        home (string): Home team name
        away (string): Away team name
        home_k (int): Home team kval
        away_k (int): Away team kval
        ind_variables (dict): Dictionary with variable name as key and coefficient as value 
    """
    home_rating = (home - (home_k * (expected_outcome[0] - expected_outcome[1])))
    away_rating = (away - (away_k * (expected_outcome[1] - expected_outcome[0])))
    return(home_rating, away_rating)


def kval_range(home, away, KVAL):
    """
    Finds the KVAL of each team in the current game from the range of kvals

    Args:
        home (int): Home team elo rating
        away (int): away team elo rating
        KVAL (list): list of kvalues for lower middle and upper tier teams

    Returns:
        team_kval (list): home and away team kvalues
    """
    team_kval = [0,0]
    n = 0
    # Determines the kval each team gets based on their elo range
    for j in (home, away):
        if j < 2100:
            team_kval[n] = KVAL[0]
        elif j in range(2100, 2400):
            team_kval[n] = KVAL[1]
        elif j > 2400:
            team_kval[n] = KVAL[2]
        n += 1

    return(team_kval)


def run_games(data, team_name, teams, cols, KVAL, ind_variables):
    """
    Takes sample data to create elo model

    Args:
        data (csv): CSV file
        team_name (string): user specified team name to graph elo rating
        teams (dict): Dictionary with variable name as key and coefficient as value
        cols (dict): dictionary of column names and locations
        KVAL (list): list of kvalue for each range
        ind_variables (dict): Dictionary with variable name as key and coefficient as value
    Returns:
        teams (dict): dictionary of team names and elo ratings
        team_vals (list): list of elo rating after each game user specified team has played
        games_played (list): list of # of games the user specified team has played
    """
    team_vals = []
    games_played = []

    # Loop through each row in sample data
    count = 0
    for i in data[1:]:
        home = i[cols['HomeTeam']]
        away = i[cols['AwayTeam']]
        winner = i[cols['FTR']]

        # get kvalues for elo ranges
        team_kval = kval_range(teams[home], teams[away], KVAL)

        # Find results if winner is the home team
        if winner == '1':
            winner_kval = team_kval[0]
            loser_kval = team_kval[1]
            # Calculate the independent variable differences
            ind_variables['Corners'] = int(i[cols['HC']]) - int(i[cols['AC']])
            ind_variables['Shots Taken'] = int(i[cols['HST']]) - int(i[cols['AST']])
            ind_variables['Foul Difference'] = int(i[cols['HF']]) - int(i[cols['AF']])
            ind_variables['Goal Difference'] = int(i[cols['FTHG']]) - int(i[cols['FTAG']])
            results = rating_update(expected_outcome(teams[home], teams[away]), teams[home], teams[away], winner_kval, loser_kval, ind_variables)
            teams[home] = results[0]
            teams[away] = results[1]

        # Find results if winner is the away team
        if winner == '0':
            winner_kval = team_kval[1]
            loser_kval = team_kval[0]
            # Calculate the independent variable differences
            ind_variables['Corners'] = int(i[cols['AC']]) - int(i[cols['HC']])
            ind_variables['Shots Taken'] = int(i[cols['AST']]) - int(i[cols['HST']])
            ind_variables['Foul Difference'] = int(i[cols['AF']]) - int(i[cols['HF']])
            ind_variables['Goal Difference'] = int(i[cols['FTAG']]) - int(i[cols['FTHG']])
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

        if home or away == team_name:
            count += 1
            team_vals.append(teams[team_name])
            games_played.append(count)

    return(teams, team_vals, games_played)


def outcome_pr(teams, home_team, away_team, gap, home_advantage):
    """
    Compares two teams

    Args:
        teams (dict): Dictionary with variable name as key and coefficient as value
        home_team (string): The home team 
        away_team (string): The away team
        gap (int): the upper and lower bound of elos for a game to be a draw
        home_advantage (int): elo rating boost for home teams
    Returns:
        string: The predicted match winner, or draw if being used in algorithm
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
        cols (dict): column names and location in the csv file (ex. HomeTeam = column # 3)
        KVAL (int): 
    Returns:
        string: Number of correct and incorrect predictions in a string
        float: The accuracy of the model
        float: The multiplier used for goal differences
    """
    real_outcome = ''
    ind_variables = {}

    # Variables for counting where our predictions are wrong
    correct = 0
    wrong = 0
    home_but_away = 0
    away_but_home = 0
    draw_but_not = 0
    notdraw_but_draw = 0

    # Runs through all games in the data set
    for i in data[1:]:
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

        # get kvalues from elo ranges
        team_kval = kval_range(teams[home], teams[away], KVAL)
        
        if winner == '1':
            winner_kval = team_kval[0]
            loser_kval = team_kval[1]
            # Calculate the independent variable differences
            ind_variables['Corners'] = int(i[cols['HC']]) - int(i[cols['AC']])
            ind_variables['Shots Taken'] = int(i[cols['HST']]) - int(i[cols['AST']])
            ind_variables['Foul Difference'] = int(i[cols['HF']]) - int(i[cols['AF']])
            ind_variables['Goal Difference'] = int(i[cols['FTHG']]) - int(i[cols['FTAG']])
            results = rating_update(expected_outcome(teams[home], teams[away]), teams[home], teams[away], winner_kval, loser_kval, ind_variables)
            teams[home] = results[0]
            teams[away] = results[1]

        if winner == '0':
            winner_kval = team_kval[1]
            loser_kval = team_kval[0]
            # Calculate the independent variable differences
            ind_variables['Corners'] = int(i[cols['AC']]) - int(i[cols['HC']])
            ind_variables['Shots Taken'] = int(i[cols['AST']]) - int(i[cols['HST']])
            ind_variables['Foul Difference'] = int(i[cols['AF']]) - int(i[cols['HF']])
            ind_variables['Goal Difference'] = int(i[cols['FTAG']]) - int(i[cols['FTHG']])
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
    """
    Args:
        teams (dict): dictionary with team names and their elo values
    Returns:
        A graph of elo distributions
    """
    data = []
    for i in teams.values():
        data.append(int(i))

    min_val = min(data)
    max_val = max(data)
    
    plt.hist(data, bins=range(min_val, max_val + 60, 60), edgecolor="k")
    plt.title("ELO Distribution")
    plt.xlabel("ELO")
    plt.ylabel("# of Teams")
    plt.show()
    

def team_growth(name, elo, games_played):
    """
    Args:
        name (string): team name that is to be graphed
        elo (list): list of teams elo after each game played
        games_played (list): list of # games played 
    Returns:
        A graph of the teams elo rating over each game theyve played
    """
    
    plt.plot(games_played, elo)
    plt.title('ELO as a function of games played %s' %name)
    plt.xlabel('Games played')
    plt.ylabel('ELO')
    plt.show()


def user_input(teams):
    """
    Takes user input for which team they want to graph

    Args:
        teams (dict): dictionary of team names
    Returns:
        team (string): team name user input 
    """
    team = input("%s\nInput a team name to see their Elo change: " %teams.keys())

    while team not in teams:
        print("Invalid team name")
        team = input("%s\nInput a team name to see their Elo change: " %teams.keys())

    return(team)


def add_teams(teams, data, cols, ELO):
    """
    Adds teams with their base ELO to dict of all teams

    Args:
        teams (dict): dictionary with team names and their elo value
        data (list): list of data from csv file
        cols (dict): dictionary of column names and their location
        ELO (int): base amount of ELO for new teams
    Returns:
        teams (Dict): dictionary of team names and their elo rating
    """
    for i in data[1:]:
        if i[cols['HomeTeam']] not in teams:
            teams[i[cols['HomeTeam']]] = ELO
        if i[cols['AwayTeam']] not in teams:
            teams[i[cols['AwayTeam']]] = ELO
    return(teams)


def data_cleaning(csv_path):
    """
    This function is used to change the value of the FTR column in the csv, home team win = 1, away team win = 0, draw = 0.5
    also removes blank rows
    Args:
        csv_path (string): csv path and name of data to be cleaned
    Returns:
        Same csv but with FTR column data cleaned
    """
    # Open the CSV and store its rows in data variable
    file_path = csv_path
    with open(file_path, 'r', newline = '') as csv_f:
        data = list(csv.reader(csv_f))

    # clean the data in the FTR column
    with open(file_path, "w", newline = '') as csv_f:
        datawriter = csv.writer(csv_f, delimiter = ',')
        rowdata = []
        for i in data:
            if i[0] != '':
                rowdata = []
                n = 0
                for j in i:
                    if n == 6:
                        if j == 'H':
                            j = 1
                        elif j == 'A':
                            j = 0
                        elif j == 'D':
                            j = 0.5
                    rowdata.append(j)
                    n += 1
                
                datawriter.writerow(rowdata)

                
def main():
    # Base ELO
    ELO = 1200
    # KVAL for elo ranges, below 2100, 2100-2400, above 2400
    KVAL = [32, 24, 16]
    # Elo range for predicting draws
    GAP = 0
    # Bonus elo for a home team
    HOME_ADVANTAGE = 200
    # List for important variables found in our regression
    ind_variables = {}

    # Sample data csv file, clean the data
    sample_path = 'C:/Users/codym/OneDrive - University of Calgary/Desktop/Econ 411 - Computer Applications/soccer data/soccer data.csv'
    data_cleaning(sample_path)
    # Holdout data to run algorithm predictions, clean the data
    holdout_path = 'C:/Users/codym/OneDrive - University of Calgary/Desktop/Econ 411 - Computer Applications/soccer data/soccer holdout data.csv'
    data_cleaning(holdout_path)
    
    # read sample data
    with open(sample_path, 'r', newline = '') as csv_f:
        data = list(csv.reader(csv_f))

    # Use column find function to store location of all columns
    cols = column_find(data)

    # Get all teams in the csv file and give them a base elo rating of 1200
    teams = {}
    teams = add_teams(teams, data, cols, ELO)
    
    # get user input for which team to be analysed
    team_name = user_input(teams)

    # Use sample data to build team elo ratings
    result = run_games(data, team_name, teams, cols, KVAL, ind_variables)

    # read holdout data
    with open(holdout_path, 'r', newline = '') as csv_f:
        data = list(csv.reader(csv_f))

    # check if there a new teams in holdout data
    teams = add_teams(teams, data, cols, ELO)

    # get predictions on holdout data
    print(ML_prediction(data, teams, cols, KVAL, GAP, HOME_ADVANTAGE))
    # graph team growth from user input
    team_growth(team_name, result[1], result[2])
    # graph elo distribution
    graph_output(teams)

    elo_results = input("Would you like to see all team Elos? (Y or N): ")
    while elo_results.upper() not in ("Y", "N"):
        print("Invalid input")
        elo_results = input("Would you like to see all team Elos? (Y or N): ".upper())
    
    if elo_results == "Y":
        # print individual elos of each team
        while len(teams) > 0:
            maxkey = max(teams, key=teams.get)
            print(maxkey + ": " + str(round(teams[maxkey], 2)))
            del teams[maxkey]


main()
