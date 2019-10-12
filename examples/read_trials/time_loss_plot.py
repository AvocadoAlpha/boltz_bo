import sys
import os
sys.path.append(os.path.abspath(os.path.join('..' ,'..', 'src')))
import pickle
import numpy as np
import importlib
import matplotlib.pyplot as plt
from list_best_performers import options, init
from hyperopt import space_eval


openF =""#sys.argv[1] # "greedy_b_n"
init(openF)
module = importlib.import_module('src.'+openF)


def my_space_eval(vals):
    mydict = {}
    for x in vals:
        if len(vals[x]) == 1:
            mydict[x] = vals[x][0]
        if len(vals[x]) > 1:
            print("ERROR1")

    return space_eval(module.space, mydict)


def normalize(a, amin):
    amax = max(a)
    for i, val in enumerate(a):
        a[i] = ((val-amin) / (amax-amin)) * 100
    return a, amax


trials = pickle.load(open('../../trials/'+ openF, "rb"))
res = trials.trials

y = [x['result']['loss'] for x in res]
first_timestamp = res[0]['book_time'].timestamp()
x, amax = normalize([round(x['refresh_time'].timestamp(), 1) for x in res], first_timestamp)


_, n = options[openF]()
from datetime import datetime
total_seconds = (datetime.fromtimestamp(amax) - res[0]['book_time']).total_seconds()
total_time = "Total Time in hours :" + str((total_seconds/60)/60) + '\n'
nOE = total_time + "Number of Evaluations :" + str(len(trials.trials)) + '\n'
best = str(my_space_eval(trials.best_trial['misc']['vals']))+" Result :"+str(trials.best_trial['result'])
fig, ax = plt.subplots()
space = module.space
text = '\n' + nOE + "Best: " + best + '\n' + module.space_str + '\n'
ax.margins(x=-0.001)
ax.set_title(openF+'\n'+text, loc='left')
ax.set_xlabel("time", fontsize=16)
ax.set_ylabel("loss", fontsize=16)
ax.set_xticks(np.arange(0, 101, 10))
fig.set_size_inches(14, 12)
ax.scatter(x, y)
plt.grid()
#plt.tight_layout()

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(0, 4), ha='center') # horizontal alignment can be left, right or center)

fig.savefig('../../plots/time-loss/'+str(openF)+'.png', dpi=200)
plt.show()
print("Figure saved in figures/")

