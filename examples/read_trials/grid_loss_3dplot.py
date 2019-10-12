import sys
import os
sys.path.append(os.path.abspath(os.path.join('..' ,'..', 'src')))
import pickle
import importlib
import matplotlib.pyplot as plt
from list_best_performers import options, init
from hyperopt import space_eval
import utils
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


openF = "dbm_grid"#sys.argv[1]
xaxe = "units1"#sys.argv[2]
module = importlib.import_module(openF)
init(openF)
header = "\ntrain: "+str(len(module.X_train)) + "\nval: " + str(len(module.X_val)) + "\ntest: "+str(len(module.X_test))+"\n"



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


trials = pickle.load(open('../trials/'+ openF, "rb"))
res = trials.trials

#y = [x['result']['loss'] for x in res]
#first_timestamp = res[0]['book_time'].timestamp()
#x, amax = normalize([round(x['refresh_time'].timestamp(), 1) for x in res], first_timestamp)


table, n = options[openF]()
sorted_table = sorted(table, key=lambda k: k['batch_size'])

x = [k["units1"] for k in table] #populate graph!
z = [k["units2"] for k in table]
y = [k['loss'] for k in table]


#from datetime import datetime
#total_seconds = (datetime.fromtimestamp(amax) - res[0]['book_time']).total_seconds()
#total_time = "Total Time in hours :" + str((total_seconds/60)/60) + '\n'
nOE =  "Number of Evaluations :" + str(len(trials.trials)) + '\n'
best = str(my_space_eval(trials.best_trial['misc']['vals']))+" Result :"+str(trials.best_trial['result'])
space = module.space
text = '\n' + nOE + "Best: " + best + '\n' + module.space_str + '\n'

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_title(openF+header+text, loc='left')


ax.invert_yaxis()


surf = ax.plot_trisurf(x, z, y)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))




fig.savefig('../../plots/grid-loss-plot3d/'+str(openF)+'.png', dpi=200, bbox_inches="tight", pad_inches=1)
plt.show()
print("Figure saved in figures/")