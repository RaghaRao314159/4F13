# expectation propagation algorithm for probit ranking
# -gtc 20/09/25
import numpy as np
from scipy.special import roots_hermitenorm
from scipy.special import ndtr as normcdf

# THIS WHOLE LOOP IS LINEAR IN THE NUMBER OF MATCHES
def exprop(games, num_players, num_its, return_msg=False, quad_degree = 64):
    """
    games : array of game outcomes, (winner, loser)
    num_players : number of players
    num_its : number of iterations of expectation prop.
    return_msg: return messages or not (default = False)
    quad_degree : number of points for Gauss-Hermite quadrature
    """
    # points for Gauss-Hermite numerical integration
    _x, _weight = roots_hermitenorm(quad_degree)

    # function to compute the mean and variance of prior * p
    def mean_var(p):
        Z = np.sum(_weight * p)
        mean_x = np.sum(_weight * _x * p) / Z 
        mean_x2 = np.sum(_weight * (_x**2) * p) / Z 
        return (mean_x, mean_x2 - mean_x**2)

    # list games for each player, X[i] = [(game_id, other_player, outcome), ...]
    # and initialize the messages 
    X = [[] for _ in range(num_players)] 
    msg = dict()
    for a, (i, j) in enumerate(games):
        # a is the match number (or index)
        X[i].append((a, j, +1))
        X[j].append((a, i, -1))
        msg[i,a] = msg[j,a] = (0.0, 1.0)  # message format is: (mean, var) 

    # message function
    F = lambda msg, y: normcdf(-y * (msg[0] - _x) / np.sqrt(1+msg[1]))
    
    # array of posterior means and variances
    posterior = np.zeros((num_players, 2))

    for _ in range(num_its):
        for i in range(num_players):
            # compute player i's posterior marginal distribution
            mrgnl = np.ones(quad_degree)
            for a, j, y_ai in X[i]:
                mrgnl *= F(msg[j,a], y_ai)

            posterior[i] = mean_var(mrgnl) # update posterior mean + variance
    
            # update the messages:
            for a, j, y_ai in X[i]:
                msg[i,a] = mean_var(mrgnl / F(msg[j,a], y_ai))

    if return_msg:
        return posterior, msg

    return posterior
