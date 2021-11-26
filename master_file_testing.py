
import csv

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
        return(outcome, 1)
    
# Function to update two teams ratings after a match
def rating_update(expected_outcome, actual_outcome, teamA, teamB, kval):
    
    teamA_rating = (teamA + kval*(actual_outcome - expected_outcome))
    teamB_rating = (teamB + kval*((1-actual_outcome) - abs(expected_outcome - 1)))

    return(teamA_rating, teamB_rating)

def run_games(data, teams):

    # Loop through each data point (game)
    print(len(data[1]))
    for i in data[1:len(data) -3]:
        home = i[2]
        away = i[3]
        winner = i[6]

        if winner == '1':
            results = rating_update(expected_outcome(teams[home], teams[away])[1], 1, teams[home], teams[away], 35)
            teams[home] = results[0]
            teams[away] = results[1]

        if winner == '0':
            results = rating_update(expected_outcome(teams[away], teams[home])[1], 0, teams[away], teams[home], 35)
            teams[home] = results[1]
            teams[away] = results[0]
        
    return(teams)

def main():
    file_path = 'C:/Users/codym/OneDrive - University of Calgary/Desktop/Econ 411 - Computer Applications/soccer data/soccer_data2.csv'
    with open(file_path, 'r', newline = '') as csv_f:
        data = list(csv.reader(csv_f))

    # Get all teams in the csv file and give them a base elo rating of 1200
    teams = {}
    for i in data[1:]:
        if i[2] not in teams:
            teams[i[2]] = 1200

    print(run_games(data, teams))

main()