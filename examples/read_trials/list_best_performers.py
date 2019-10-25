import sys
import os
sys.path.append(os.path.abspath(os.path.join('..' ,'..', 'src')))
import pickle
import numpy as np
import importlib
from hyperopt import space_eval

flag_grid_or_bo = True
x = []
y = []
module = None
res = None
trials = None
openF = ""
def init(openF1):
    global openF
    openF = openF1
    global module
    module = importlib.import_module("examples."+openF1)
    global trials
    trials = pickle.load(open('../trials/'+ openF1, "rb"))
    global res
    res = trials.trials
    global y
    y = [x['result']['loss'] for x in res]
    global x
    x = [x['misc']['tid'] for x in res]
    print(openF)
    table, _ = options[openF]()
    sorted_table = sorted(table, key=lambda k: k['loss'])

    print("20 Best Performers")
    if flag_grid_or_bo and "fixed" in openF:
        print("units are uniform values between 0 and 100 representing fractions")
    for x in sorted_table[0:20]:
        print(str(x))

def my_space_eval(vals):
    mydict = {}
    for x in vals:
        if len(vals[x]) == 1:
            mydict[x] = vals[x][0]
        if len(vals[x]) > 1:
            print("ERROR1")

    return space_eval(module.space, mydict)

def pars (vals):
    listOf = []
    y = module.layer_options[vals['layers'][0]]
    pre = 784

    for x in range(0, y):
        pre = int(np.ceil(pre * vals['n_nodes_' + str(y) + '_' + str(x)][0]))
        listOf.append(pre)

    return tuple(listOf)



def nodes_batch():
    n = [(my_space_eval(x['misc']['vals'])['units1'], my_space_eval(x['misc']['vals'])['batch_size']) for x in res]
    n = [(int(x[0]), x[1]) for x in n]
    table = [{"loss": x['result']['loss'], "units1": my_space_eval(x['misc']['vals'])['units1'],
              "batch_size": my_space_eval(x['misc']['vals'])['batch_size']} for x in res]
    return table, n

#for gidsearch
def fixed_3():
    n = [(x['result']['loss'], my_space_eval(x['misc']['vals'])['units1'], (my_space_eval(x['misc']['vals'])['units2']),
          my_space_eval(x['misc']['vals'])['batch_size']) for x in res]
    n = [(x[0], x[1], x[2], x[3]) for x in n]
    table = [{"loss": x[0], "units1": x[1], "units2": x[2], "batch_size": x[3]} for x in n]
    return table, n


def fixed_5():
    n = [(x['result']['loss'], my_space_eval(x['misc']['vals'])['units1'], my_space_eval(x['misc']['vals'])['units2'],
          my_space_eval(x['misc']['vals'])['units3'], my_space_eval(x['misc']['vals'])['batch_size']) for x in res]
    n = [(x[0], x[1], x[2], x[3], x[4]) for x in n]
    table = [{"loss": x[0], "units1": x[1], "units2": x[2], "units3": x[3], "batch_size": x[4]} for x in n]
    return table, n


def non_greedy():
    n = [(pars(x['misc']['vals']), my_space_eval(x['misc']['vals'])['batch_size']) for x in res]
    table = [{"loss": x['result']['loss'], "units": pars(x['misc']['vals']),
              "batch_size": my_space_eval(x['misc']['vals'])['batch_size']} for x in res]
    return table, n

"""
#tree parzen
def fixed_3():
    n = [(x['result']['loss'], my_space_eval(x['misc']['vals'])['units1'], (my_space_eval(x['misc']['vals'])['units2']),
          my_space_eval(x['misc']['vals'])['batch_size']) for x in res]
    n = [(x[0], int(np.ceil(784 * x[1])), int(np.ceil(int(np.ceil(784 * x[1])) * x[2])), x[3]) for x in n]
    table = [{"loss": x[0], "units1": x[1], "units2": x[2], "batch_size": x[3]} for x in n]
    return table, n


def fixed_5():
    n = [(x['result']['loss'], my_space_eval(x['misc']['vals'])['units1'], my_space_eval(x['misc']['vals'])['units2'],
          my_space_eval(x['misc']['vals'])['units3'], my_space_eval(x['misc']['vals'])['batch_size']) for x in res]
    n = [(x[0], int(np.ceil(784 * x[1])), int(np.ceil(int(np.ceil(784 * x[1]))) * x[2]),
          int(np.ceil(int(np.ceil(int(np.ceil(784 * x[1]))) * x[2])) * x[3]), x[4]) for x in n]
    table = [{"loss": x[0], "units1": x[1], "units2": x[2], "units3": x[3], "batch_size": x[4]} for x in n]
    return table, n
"""



options = { "grid_batch" : nodes_batch,
            "grid_nodes" :nodes_batch,
            "rbm_nodes_grid": nodes_batch,
            "rbm_nodes_grid_2": nodes_batch,
            "grid_nodes_100" :nodes_batch,
            "grid_nodes_3" :nodes_batch,
            "grid_nodes_keras":nodes_batch,
            "grid_nodes_keras_no_reg":nodes_batch,
            "greedy": nodes_batch,
            "greedy_b_n":nodes_batch,
            "greedy_2": nodes_batch,
            "greedy_batch": nodes_batch,
            "greedy_nodes": nodes_batch,
            "fixed_3": fixed_3,
            "fixed_3_2": fixed_3,
            "fixed_3_kl1": fixed_3,
            "fixed_3_kl2": fixed_3,
            "fixed_5": fixed_5,
            "non_greedy_3": non_greedy,
            "non_greedy_5": non_greedy,
            "dbm_fixed_5": fixed_5,
            "dbm_grid": fixed_3,
            "dbm_non_greedz_5":non_greedy,
            }








