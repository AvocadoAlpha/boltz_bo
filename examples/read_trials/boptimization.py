import sys
import pickle
from hyperopt import space_eval, fmin, Trials, tpe, rand
from keras.datasets import mnist
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
import exhaustive_search
import os
"""
def run_trials(script_name, space, objective):

    trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 5  # initial max_trials. put something small to not have to wait

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("trials/"+script_name, "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=max_trials)

    print("Best:", best)
    print("SpaceEval" + str(space_eval(space, trials.argmin)))

    # save the trials object
    with open("trials/" + script_name, "wb") as f:
        pickle.dump(trials, f)

"""
def run_trials_grid(script_name, space, objective):

    exhaustive_search.validate_space_exhaustive_search(space)
    trials = Trials()
    best = fmin(objective, space, algo=exhaustive_search.partial(exhaustive_search.suggest, nbMaxSucessiveFailures=1000), trials=trials, max_evals=1000000)

    print("Best:", best)
    print("SpaceEval" + str(space_eval(space, trials.argmin)))

    # save the trials object
    with open("trials/"+script_name+"_grid", "wb") as f:
        pickle.dump(trials, f)

def run_trials_grid_2(script_name, space, objective):

    exhaustive_search.validate_space_exhaustive_search(space)
    trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 5  # initial max_trials. put something small to not have to wait

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("trials/"+script_name, "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    best = fmin(objective, space, algo=exhaustive_search.partial(exhaustive_search.suggest, nbMaxSucessiveFailures=1000), trials=trials, max_evals=max_trials)

    print("Best:", best)
    print("SpaceEval" + str(space_eval(space, trials.argmin)))

    # save the trials object
    with open("trials/" + script_name, "wb") as f:
        pickle.dump(trials, f)