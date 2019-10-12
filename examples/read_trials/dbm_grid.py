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
import env
from boltzmann_machines import DBM
from boltzmann_machines.rbm import BernoulliRBM


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
                        model_path=args.rbm1_dirpath)
    rbm1.fit(X)
    return rbm1

def make_rbm2(Q, args):

    print "\nTraining RBM #2 ...\n\n"

    epochs = args.epochs[1]
    n_every = args.increase_n_gibbs_steps_every

    n_gibbs_steps = np.arange(args.n_gibbs_steps[1],
                              args.n_gibbs_steps[1] + epochs / n_every)
    learning_rate = args.lr[1] / np.arange(1, 1 + epochs / n_every)
    n_gibbs_steps = np.repeat(n_gibbs_steps, n_every)
    learning_rate = np.repeat(learning_rate, n_every)

    rbm2 = BernoulliRBM(n_visible=args.n_hiddens[0],
                        n_hidden=args.n_hiddens[1],
                        W_init=0.005,
                        vb_init=0.,
                        hb_init=0.,
                        n_gibbs_steps=n_gibbs_steps,
                        learning_rate=learning_rate,
                        momentum=[0.5] * 5 + [0.9],
                        max_epoch=max(args.epochs[1], n_every),
                        batch_size=args.batch_size[1],
                        l2=args.l2[1],
                        sample_h_states=True,
                        sample_v_states=True,
                        sparsity_cost=0.,
                        dbm_last=True,  # !!!
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
                        model_path=args.rbm2_dirpath)
    rbm2.fit(Q)
    return rbm2

def make_dbm((X_train, X_val), rbms, (Q, G), args):
    print "\nMake dbm #2 ...\n\n"
    dbm = DBM(rbms=rbms,
              n_particles=args.n_particles,
              v_particle_init=X_train[:args.n_particles].copy(),
              h_particles_init=(Q[:args.n_particles].copy(),
                                G[:args.n_particles].copy()),
              n_gibbs_steps=args.n_gibbs_steps[2],
              max_mf_updates=args.max_mf_updates,
              mf_tol=args.mf_tol,
              learning_rate=np.geomspace(args.lr[2], 5e-6, 400),
              momentum=np.geomspace(0.5, 0.9, 10),
              max_epoch=args.epochs[2],
              batch_size=args.batch_size[2],
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
              model_path=args.dbm_dirpath)
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

# common for RBMs and DBM
parser.add_argument('--n-hiddens', type=int, default=(200, 30), metavar='N', nargs='+',
                    help='numbers of hidden units')
parser.add_argument('--n-gibbs-steps', type=int, default=(1, 1, 1), metavar='N', nargs='+',
                    help='(initial) number of Gibbs steps for CD/PCD')
parser.add_argument('--lr', type=float, default=(0.05, 0.01, 2e-3), metavar='LR', nargs='+',
                    help='(initial) learning rates')
parser.add_argument('--epochs', type=int, default=(64, 128, 512), metavar='N', nargs='+',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=(48, 48, 100), metavar='B', nargs='+',
                    help='input batch size for training, `--n-train` and `--n-val`' + \
                         'must be divisible by this number (for DBM)')
parser.add_argument('--l2', type=float, default=(1e-3, 2e-4, 1e-7), metavar='L2', nargs='+',
                    help='L2 weight decay coefficients')
parser.add_argument('--random-seed', type=int, default=(1337, 1111, 2222), metavar='N', nargs='+',
                    help='random seeds for models training')

# save dirpaths
parser.add_argument('--rbm1-dirpath', type=str, default='../models/dbm_mnist_rbm1/', metavar='DIRPATH',
                    help='directory path to save RBM #1')
parser.add_argument('--rbm2-dirpath', type=str, default='../models/dbm_mnist_rbm2/', metavar='DIRPATH',
                    help='directory path to save RBM #2')
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
parser.add_argument('--sparsity-target', type=float, default=(0.2, 0.1), metavar='T', nargs='+',
                    help='desired probability of hidden activation')
parser.add_argument('--sparsity-cost', type=float, default=(1e-4, 5e-5), metavar='C', nargs='+',
                    help='controls the amount of sparsity penalty')
parser.add_argument('--sparsity-damping', type=float, default=0.9, metavar='D',
                    help='decay rate for hidden activations probs')


# parse and check params
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
script_name = os.path.basename(__file__).split('.')[0]
X_train, X_val, X_test = generate_data.generate_data_medium_2()
X = np.concatenate((X_train, X_val))

space = {
    'units1': hp.quniform('units1', 0, 100, 5), #implementation of hq.uniform is weird see github.com/hyperopt/hyperopt/issues/321
    'units2': hp.quniform('units2', 0, 100, 5), #implementation of hq.uniform is weird see github.com/hyperopt/hyperopt/issues/321
    'batch_size': hp.choice('batch_size', [128])
    }

space_str = """
space = {
    'units1': hp.quniform('units1', 0, 100, 5), 
    'units2': hp.quniform('units2', 0, 100, 5), 
    'batch_size': hp.choice('batch_size', [128])
    }"""

def objective(params):
    for x in params.keys(): # if "units1":0 add one -> units1:1
        if params[x] == 0:
            params[x] = 5
    K.clear_session()

    layer1 = int(np.ceil((params['units1'] / 100) * 784))
    layer2 = max(int(np.ceil((params['units2'] / 100) * layer1)), 24)
    args.n_hiddens = (layer1, layer2)
    print("Params :")
    print("Params :" + str(layer1))
    print("Params :" + str(layer2))
    # pre-train RBM #1
    rbm1 = make_rbm1(X, args)

    # freeze RBM #1 and extract features Q = p_{RBM_1}(h|v=X)
    Q = rbm1.transform(X)


    # pre-train RBM #2
    rbm2 = make_rbm2(Q, args)

    # freeze RBM #2 and extract features G = p_{RBM_2}(h|v=Q)
    G = rbm2.transform(Q)

    # jointly train DBM
    dbm = make_dbm((X_train, X_val), (rbm1, rbm2), (Q, G), args)


    predictions = dbm.reconstruct(X_test)
    print(predictions)

    loss = tf.keras.backend.sum(mean_squared_error(tf.convert_to_tensor(X_test), tf.convert_to_tensor(predictions)))
    sess = tf.Session()
    score = round(sess.run(loss) / len(X_test), 4)
    print(score)
    return {'loss': score, 'status': STATUS_OK}

if __name__ == "__main__":
    while True:
        boptimization.run_trials_grid_2(script_name, space, objective)


