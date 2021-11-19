# Function to estimate outcome of two teams given their rating
# Takes two ratings as parameters
# Returns the expected outcome of their game, winning team and expected probability of winning
# https://en.wikipedia.org/wiki/Elo_rating_system#Theory
def expected_outcome(ratingA, ratingB):
    outcome = ''
    expectedA = (1/(1+10**((ratingB - ratingA)/400)))
    expectedB = (1/(1+10**((ratingA - ratingB)/400)))

    if expectedA > expectedB:
        outcome = 'Team A'
        return(outcome, expectedA)
    
    if expectedA < expectedB:
        outcome = 'Team B'
        return(outcome, expectedB)
    
    if expectedA == expectedB:
        outcome = 'Draw'
        return(outcome)
    
# Function to update two teams ratings after a match
def rating_update(expected_outcome, actual_outcome, teamA, teamB, kval):
    n = 0
    for i in expected_outcome:
        teamA_rating = (teamA + kval*(actual_outcome[n] - i))
        teamB_rating = (teamB + kval*((1-actual_outcome[n]) - abs(i - 1)))

        n += 1
    return(teamA_rating, teamB_rating)

# Function to run each game and update the teams elo rating
# Takes no parameters
# Returns no values
def run_games():

    # Loop through each data point (game)
    for i in data[1:]:
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
