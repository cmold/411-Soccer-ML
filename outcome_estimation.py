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
    if expectedA < expectedB:
        outcome = 'Team B'
    if expectedA == expectedB:
        outcome = 'Draw'

    return(outcome, max([expectedA, expectedB]))

# Function to update two teams ratings after a match
def rating_update(expected_outcome, actual_outcome, teamA, teamB, kval):
    n = 0
    for i in expected_outcome:
        teamA_rating = (teamA + kval*(actual_outcome[n] - i))
        teamB_rating = (teamB + kval*((1-actual_outcome[n]) - abs(i - 1)))

        n += 1
    return(teamA_rating, teamB_rating

print(expected_outcome(1900, 1500))
