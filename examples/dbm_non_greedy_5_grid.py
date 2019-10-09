#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train 2-layer Bernoulli DBM on MNIST dataset with pre-training.
Hyper-parameters are similar to those in MATLAB code [1].
Some of them were changed for more efficient computation on GPUs,
another ones to obtain more stable learning (lesser number of "died" units etc.)
RBM #2 trained with increasing k in CD-k and decreasing learning rate
over time.

Per sample validation mean reconstruction error for DBM (mostly) monotonically
decreases during training and is about 5.27e-3 at the end.

The training took approx. 9 + 55 + 185 min = 4h 9m on GTX 1060.

After the model is trained, it is discriminatively fine-tuned.
The code uses early stopping so max number of MLP epochs is often not reached.
It achieves 1.32% misclassification rate on the test set.

Note that DBM is trained without centering.

Links
-----
[1] http://www.cs.toronto.edu/~rsalakhu/DBM.html
"""

print __doc__
import env
from keras import backend as K
import generate_data
import boptimization
from hyperopt import hp, STATUS_OK
import os
import argparse
import numpy as np
import tensorflow as tf
from keras.losses import mean_squared_error
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval
from keras.losses import binary_crossentropy
import tensorflow as tf
from keras import backend as K
import os
import pickle

import os
import argparse
import numpy as np

import env
from boltzmann_machines import DBM
from boltzmann_machines.rbm import BernoulliRBM
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

script_name = os.path.basename(__file__).split('.')[0]


def make_rbm1(X, args):
    print "\nTraining RBM #1 ...\n\n"
    rbm1 = BernoulliRBM(n_visible=784,
                        n_hidden=args.n_hiddens[0],
                        W_init=0.001,
                        vb_init=0.,
                        hb_init=0.,
                        n_gibbs_steps=args.n_gibbs_steps[0],
                        learning_rate=args.lr[0],
                        momentum=[0.5] * 5 + [0.9],
                        max_epoch=args.epochs[0],
                        batch_size=args.batch_size[0],
                        l2=args.l2[0],
                        sample_h_states=True,
                        sample_v_states=True,
                        sparsity_cost=0.,
                        dbm_first=True,  # !!!
                        metrics_config=dict(
                            msre=True,
                            pll=True,
                            train_metrics_every_iter=500,
                        ),
                        verbose=True,
                        display_filters=30,
                        display_hidden_activations=24,
                        v_shape=(28, 28),
                        random_seed=args.random_seed[0],
                        dtype='float32',
                        tf_saver_params=dict(max_to_keep=1),
                        model_path=args.model_path + 'rbm_1/')
    rbm1.fit(X)
    return rbm1


def make_rbm_n(Q, args, i):  # is for which layer to make start with 2
    i = i - 1  # for index in arguments
    print "\nTraining RBM " + str(i + 1) + " ...\n\n"

    epochs = args.epochs[1]
    n_every = args.increase_n_gibbs_steps_every

    n_gibbs_steps = np.arange(args.n_gibbs_steps[1],
                              args.n_gibbs_steps[1] + epochs / n_every)
    learning_rate = args.lr[1] / np.arange(1, 1 + epochs / n_every)
    n_gibbs_steps = np.repeat(n_gibbs_steps, n_every)
    learning_rate = np.repeat(learning_rate, n_every)

    rbm2 = BernoulliRBM(n_visible=args.n_hiddens[i - 1],
                        n_hidden=args.n_hiddens[i],
                        W_init=0.005,
                        vb_init=0.,
                        hb_init=0.,
                        n_gibbs_steps=n_gibbs_steps,
                        learning_rate=learning_rate,
                        momentum=[0.5] * 5 + [0.9],
                        max_epoch=max(args.epochs[i], n_every),
                        batch_size=args.batch_size[i],
                        l2=args.l2[1],
                        sample_h_states=True,
                        sample_v_states=True,
                        sparsity_cost=0.,
                        dbm_last=args.dbm_last,  # !!!
                        metrics_config=dict(
                            msre=True,
                            pll=True,
                            train_metrics_every_iter=500,
                        ),
                        verbose=True,
                        display_filters=0,
                        display_hidden_activations=24,
                        random_seed=args.random_seed[1],
                        dtype='float32',
                        tf_saver_params=dict(max_to_keep=1),
                        model_path=args.model_path + 'rbm_' + str(i + 1)+"/")
    rbm2.fit(Q)
    return rbm2


def make_dbm((X_train, X_val), rbms, transforms, args):
    print "\nTraining DBM ...\n\n"
    dbm = DBM(rbms=rbms,
              n_particles=args.n_particles,
              v_particle_init=X_train[:args.n_particles].copy(),
              h_particles_init=[xz[:args.n_particles].copy() for xz in transforms],
              n_gibbs_steps=args.n_gibbs_steps[3],
              max_mf_updates=args.max_mf_updates,
              mf_tol=args.mf_tol,
              learning_rate=np.geomspace(args.lr[2], 5e-6, 400),
              momentum=np.geomspace(0.5, 0.9, 10),
              max_epoch=args.epochs[3],
              batch_size=args.batch_size[3],
              l2=args.l2[2],
              max_norm=args.max_norm,
              sample_v_states=True,
              sample_h_states=(True, True),
              sparsity_target=args.sparsity_target,
              sparsity_cost=args.sparsity_cost,
              sparsity_damping=args.sparsity_damping,
              train_metrics_every_iter=400,
              val_metrics_every_epoch=2,
              random_seed=args.random_seed[2],
              verbose=True,
              display_filters=10,
              display_particles=20,
              v_shape=(28, 28),
              dtype='float32',
              tf_saver_params=dict(max_to_keep=1),
              model_path=args.model_path + 'dbm/')
    dbm.fit(X_train, X_val)
    return dbm


# training settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# general/data
parser.add_argument('--gpu', type=str, default='0', metavar='ID',
                    help="ID of the GPU to train on (or '' to train on CPU)")
parser.add_argument('--n-train', type=int, default=59000, metavar='N',
                    help='number of training examples')
parser.add_argument('--n-val', type=int, default=1000, metavar='N',
                    help='number of validation examples')

# RBM #2 related
parser.add_argument('--increase-n-gibbs-steps-every', type=int, default=20, metavar='I',
                    help='increase number of Gibbs steps every specified number of epochs for RBM #2')
parser.add_argument('--dbm_last', type=bool, metavar='dbm')

# common for RBMs and DBM
parser.add_argument('--n-hiddens', type=int, default=[], metavar='N', nargs='+',
                    help='numbers of hidden units')
parser.add_argument('--n-gibbs-steps', type=int, default=(1, 1, 1, 1, 1), metavar='N', nargs='+',
                    help='(initial) number of Gibbs steps for CD/PCD')
parser.add_argument('--lr', type=float, default=(0.05, 0.01, 2e-3, 4e-4), metavar='LR', nargs='+',
                    help='(initial) learning rates')
parser.add_argument('--epochs', type=int, default=(30, 30, 30, 10), metavar='N', nargs='+',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=(480, 480, 480, 1000), metavar='B', nargs='+',
                    help='input batch size for training, `--n-train` and `--n-val`' + \
                         'must be divisible by this number (for DBM)')
parser.add_argument('--l2', type=float, default=(1e-3, 2e-4, 1e-7), metavar='L2', nargs='+',
                    help='L2 weight decay coefficients')
parser.add_argument('--random-seed', type=int, default=(1337, 1111, 2222), metavar='N', nargs='+',
                    help='random seeds for models training')

# save dirpaths
parser.add_argument('--model_path', type=str, default='../models/', metavar='DIRPATH',
                    help='directory path to save')

# save dirpaths
parser.add_argument('--rbm1-dirpath', type=str, default='../models/dbm_mnist_rbm1/', metavar='DIRPATH',
                    help='directory path to save RBM #1')
parser.add_argument('--rbm2-dirpath', type=str, default='../models/dbm_mnist_rbm2/', metavar='DIRPATH',
                    help='directory path to save RBM #2')
parser.add_argument('--rbm3-dirpath', type=str, default='../models/dbm_mnist_rbm3/', metavar='DIRPATH',
                    help='directory path to save RBM #3')
parser.add_argument('--dbm-dirpath', type=str, default='../models/dbm_mnist/', metavar='DIRPATH',
                    help='directory path to save DBM')

# DBM related
parser.add_argument('--n-particles', type=int, default=100, metavar='M',
                    help='number of persistent Markov chains')
parser.add_argument('--max-mf-updates', type=int, default=50, metavar='N',
                    help='maximum number of mean-field updates per weight update')
parser.add_argument('--mf-tol', type=float, default=1e-7, metavar='TOL',
                    help='mean-field tolerance')
parser.add_argument('--max-norm', type=float, default=6., metavar='C',
                    help='maximum norm constraint')
parser.add_argument('--sparsity-target', type=float, default=(0.2, 0.1, 0.05, 0.0025, 0.00173), metavar='T', nargs='+',
                    help='desired probability of hidden activation')
parser.add_argument('--sparsity-cost', type=float, default=(1e-4, 5e-5, 3e-6, 2e-7, 1e-8), metavar='C', nargs='+',
                    help='controls the amount of sparsity penalty')
parser.add_argument('--sparsity-damping', type=float, default=0.9, metavar='D',
                    help='decay rate for hidden activations probs')

# parse and check params
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
script_name = os.path.basename(__file__).split('.')[0]
X_train, X_val, X_test = generate_data.generate_data_medium_2()
X = np.concatenate((X_train, X_val))

# prepare data (load + scale + split)
print "\nPreparing data ...\n\n"


# Hyperopt
def node_params(n_layers):
    # define the parameters that are conditional on the number of layers here
    # in this case the number of nodes in each layer
    params = {}
    for n in range(n_layers):
        params['n_nodes_layer_{}'.format(n)] = hp.quniform('n_nodes_{}_{}'.format(n_layers, n), 0, 100, 20)
    return params


# list of the number of layers you want to consider
layer_options = [3]

# dynamically build the space based on the possible number of layers
space = {'choice': hp.choice('layers', [node_params(n) for n in layer_options]),

         # 'batch_size': hp.choice('batch_size', [10, 32, 64, 128, 256, 512, 1024])
         }
space_str = """space = {'units1': hp.quniform('units1', 0, 784, 1),\n                'batch_size': hp.choice('batch_size', [10, 32, 64, 128, 256, 512, 1024])}"""


def objective(params):
    print('Params testing: ', params)
    print('\n ')


    for x in params.keys():
        for y in params[x].keys():# if "units1":0 add one -> units1:1
            if params[x][y] == 0:
                params[x][y] = 5
    K.clear_session()

    layersAndNodes = list(params['choice'].values())  # len is number of layers
    print("Layers :" + str(len(layersAndNodes)))
    next_layer = int(np.ceil((layersAndNodes[0] / 100) * 784))
    args.n_hiddens.append(next_layer)
    rbm1 = make_rbm1(X, args)

    transforms = []
    rbm_n = []
    rbm_n.append(rbm1)
    transforms.append(rbm1.transform(X))
    start = 2  # counter for n in rbm_n
    counter = 0

    for x1 in layersAndNodes[1:]:
        print("Layer :" + str(start))
        next_layer = max(int(np.ceil(x1 / 100 * next_layer)), 25)
        print("Nodes :" + str(next_layer))
        args.n_hiddens.append(next_layer)
        if x1 == layersAndNodes[len(layersAndNodes) - 1]:
            args.dbm_last = True
        else:
            args.dbm_last = False

        rbm_n.append(make_rbm_n(transforms[counter], args, start))
        transforms.append(rbm_n[counter + 1].transform(transforms[counter]))

        counter = counter + 1
        start = start + 1

    # jointly train DBM
    dbm = make_dbm((X_train, X_val), rbm_n, transforms, args)

    preds = dbm.reconstruct(X_test)

    loss = tf.keras.backend.sum(binary_crossentropy(tf.convert_to_tensor(X_test), tf.convert_to_tensor(preds)))
    sess = tf.Session()
    score = round(sess.run(loss) / len(X_test), 4)

    return {'loss': score, 'status': STATUS_OK}


while True:
    boptimization.run_trials_grid_2(script_name, space, objective)

