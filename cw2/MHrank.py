# Metropolis-Hastings MCMC algorithm for sampling skills in the probit rank model
# -gtc 20/09/2025
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

def MH_sample(games, num_players, num_its, return_accept_ratio=False, std =1.4):

    # pre-process data:
    # array of games for each player, X[i] = [(other_player, outcome), ...]
    X = [[] for _ in range(num_players)] 
    for a, (i,j) in enumerate(games):
        X[i].append((j, +1))  # player i beat player j
        X[j].append((i, -1))  # player j lost to payer i
    for i in range(num_players):
        X[i] = np.array(X[i])

    # array that will contain skill samples
    skill_samples = np.zeros((num_players, num_its))
    
    # track acceptance ratios per iteration (store the actual ratios, not just counts)
    acceptance_ratios_per_player = np.zeros((num_players, num_its))

    w = np.zeros(num_players)  # skill for each player

    for itr in tqdm(range(num_its)):
        for i in range(num_players):
            j, outcome = X[i].T

            # current local log-prob 
            lp1 = norm.logpdf(w[i]) + np.sum(norm.logcdf(outcome*(w[i]-w[j])))

            # proposed new skill and log-prob
            new_skill = w[i] + np.random.normal(0, std)
            lp2 = norm.logpdf(new_skill) + np.sum(norm.logcdf(outcome*(new_skill - w[j])))

            # accept or reject move:
            u = np.log(np.random.uniform(0,1))
            acceptance_log_ratio = min(lp2 - lp1, 0)
            
            # Store the actual acceptance ratio (exp of log ratio)
            acceptance_ratios_per_player[i, itr] = np.exp(acceptance_log_ratio)

            if u < acceptance_log_ratio:
                # accept move
                w[i] = new_skill

        skill_samples[:, itr] = w

    if return_accept_ratio:
        # Return mean acceptance ratio per iteration across all players
        acceptance_ratios = np.mean(acceptance_ratios_per_player, axis=0)
        return skill_samples, acceptance_ratios
    return skill_samples

