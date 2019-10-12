# %load read_trials.py
import pickle
import numpy as np
import importlib
import matplotlib.pyplot as plt
from hyperopt import space_eval
openF = 'rbm_nodes_grid'
#module = importlib.import_module('src.'+openF)

trials = pickle.load(open('../trials/'+ openF, "rb"))
print("Length :" + str(len(trials.trials)))
print("Trials :" + str(trials.trials))
print("Best Trials :" + str(trials.best_trial))

