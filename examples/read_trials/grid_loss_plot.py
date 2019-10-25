import sys
import os
sys.path.append(os.path.abspath(os.path.join('..' ,'..', 'src')))
import pickle
import importlib

from list_best_performers import options, init
from hyperopt import space_eval
import utils
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
openF = "rbm_nodes_grid_2"#sys.argv[1]

xaxe = "units1"#sys.argv[2]
module = importlib.import_module("examples."+openF)
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
#sorted_table = sorted(table, key=lambda k: k['batch_size'])


x = [t[xaxe] for t in table] #populate graph!
y = [t['loss'] for t in table]
"""
x = [x['units1'] for x in table]
y = [x['loss'] for x in table]
"""




from datetime import datetime
#total_seconds = (datetime.fromtimestamp(amax) - res[0]['book_time']).total_seconds()
#total_time = "Total Time in hours :" + str((total_seconds/60)/60) + '\n'
nOE = "Number of Evaluations :" + str(len(trials.trials)) + '\n'
best = str(my_space_eval(trials.best_trial['misc']['vals']))+" Result :"+str(trials.best_trial['result'])
fig, ax = plt.subplots()
space = module.space
text = '\n' + nOE + "Best: " + best + '\n' + module.space_str + '\n'
#ax.margins(x=-0.001)
ax.set_title(openF+header+text, loc='left')
ax.set_xlabel(xaxe, fontsize=16)
ax.set_ylabel("loss", fontsize=16)
#ax.set_xticks(np.arange(0, 101, 10))
fig.set_size_inches(12, 12)
ax.scatter(x, y, s=12)
plt.grid()
#plt.tight_layout()

"""
for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(0, 4), ha='center') # horizontal alignment can be left, right or center)
"""
fig.savefig('../../plots/grid-loss-plot/'+str(openF)+'.png', dpi=200, bbox_inches='tight')
plt.show()
print("Figure saved in figures/")