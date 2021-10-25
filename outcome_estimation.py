# Function to estimate outcome of two teams given their rating
# Takes two ratings as parameters
# Returns the expected outcome of their game, winning team and expected probability of winning
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

print(expected_outcome(1900, 1500))
