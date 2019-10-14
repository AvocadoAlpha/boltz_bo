#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Bernoulli-Bernoulli RBM on MNIST dataset and use for classification.

Momentum is initially 0.5 and gradually increases to 0.9.
Training time is approx. 2.5 times faster using single-precision rather than double
with negligible difference in reconstruction error, pseudo log-likelihood is slightly
more noisy at the beginning of training though.

Per sample validation pseudo log-likelihood is -0.08 after 28 epochs and -0.017 after 110
epochs. It still slightly underfitting at that point, though (free energy gap at the end
of training is -1.4 < 0). Average validation mean reconstruction error monotonically
decreases during training and is about 7.39e-3 at the end.

The training took approx. 38 min on GTX 1060.

After the model is trained, it is discriminatively fine-tuned.
The code uses early stopping so max number of MLP epochs is often not reached.
It achieves 1.27% misclassification rate on the test set.
"""
from __future__ import division
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
from boltzmann_machines.rbm import BernoulliRBM, logit_mean




def make_rbm(X_train, X_val, args):
    print "\nTraining model ...\n\n"
    rbm = BernoulliRBM(n_visible=784,
                       n_hidden=args.n_hidden,
                       W_init=args.w_init,
                       vb_init=logit_mean(X_train) if args.vb_init else 0.,
                       hb_init=args.hb_init,
                       n_gibbs_steps=args.n_gibbs_steps,
                       learning_rate=args.lr,
                       momentum=np.geomspace(0.5, 0.9, 8),
                       max_epoch=args.epochs,
                       batch_size=args.batch_size,
                       l2=args.l2,
                       sample_v_states=args.sample_v_states,
                       sample_h_states=True,
                       dropout=args.dropout,
                       sparsity_target=args.sparsity_target,
                       sparsity_cost=args.sparsity_cost,
                       sparsity_damping=args.sparsity_damping,
                       metrics_config=dict(
                           msre=True,
                           pll=True,
                           feg=True,
                           train_metrics_every_iter=1000,
                           val_metrics_every_epoch=2,
                           feg_every_epoch=4,
                           n_batches_for_feg=50,
                       ),
                       verbose=True,
                       display_filters=30,
                       display_hidden_activations=24,
                       v_shape=(28, 28),
                       random_seed=args.random_seed,
                       dtype=args.dtype,
                       tf_saver_params=dict(max_to_keep=1))
    rbm.fit(X_train, X_val)
    return rbm


# training settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# general/data

parser.add_argument('--gpu', type=str, default='0', metavar='ID',
                    help="ID of the GPU to train on (or '' to train on CPU)")
parser.add_argument('--n-train', type=int, default=55000, metavar='N',
                    help='number of training examples')
parser.add_argument('--n-val', type=int, default=5000, metavar='N',
                    help='number of validation examples')
parser.add_argument('--data-path', type=str, default='../data/', metavar='PATH',
                    help='directory for storing augmented data etc.')

# RBM related
parser.add_argument('--n-hidden', type=int, default=1024, metavar='N',
                    help='number of hidden units')
parser.add_argument('--w-init', type=float, default=0.01, metavar='STD',
                    help='initialize weights from zero-centered Gaussian with this standard deviation')
parser.add_argument('--vb-init', action='store_false',
                    help='initialize visible biases as logit of mean values of features' + \
                         ', otherwise (if enabled) zero init')
parser.add_argument('--hb-init', type=float, default=0., metavar='HB',
                    help='initial hidden bias')
parser.add_argument('--n-gibbs-steps', type=int, default=1, metavar='N', nargs='+',
                    help='number of Gibbs updates per weights update or sequence of such (per epoch)')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR', nargs='+',
                    help='learning rate or sequence of such (per epoch)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=10, metavar='B',
                    help='input batch size for training')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2',
                    help='L2 weight decay coefficient')
parser.add_argument('--sample-v-states', action='store_true',
                    help='sample visible states, otherwise use probabilities w/o sampling')
parser.add_argument('--dropout', type=float, metavar='P',
                    help='probability of visible units being on')
parser.add_argument('--sparsity-target', type=float, default=0.1, metavar='T',
                    help='desired probability of hidden activation')
parser.add_argument('--sparsity-cost', type=float, default=1e-5, metavar='C',
                    help='controls the amount of sparsity penalty')
parser.add_argument('--sparsity-damping', type=float, default=0.9, metavar='D',
                    help='decay rate for hidden activations probs')
parser.add_argument('--random-seed', type=int, default=1337, metavar='N',
                    help="random seed for model training")
parser.add_argument('--dtype', type=str, default='float32', metavar='T',
                    help="datatype precision to use")


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
script_name = os.path.basename(__file__).split('.')[0]

X_train, X_val, X_test = generate_data.generate_data_medium_2()

space = {
    'units1': hp.quniform('units1', 0, 2000, 10), #implementation of hq.uniform is weird see github.com/hyperopt/hyperopt/issues/321
    'batch_size': hp.choice('batch_size', [128])
    }

space_str = """
space = {
    'units1': hp.quniform('units1', 0, 754, 10), 
    'batch_size': hp.choice('batch_size', [128])
    }"""


def objective(params):
    for x in params.keys(): # if "units1":0 add one -> units1:1
            params[x] = params[x] + 30
    K.clear_session()

    args.n_hidden = int(params['units1'])
    # train and save the RBM model
    rbm = make_rbm(X_train, X_val, args)

    predictions = rbm.reconstruct(X_test)
    print(predictions)

    loss = tf.keras.backend.sum(mean_squared_error(tf.convert_to_tensor(X_test), tf.convert_to_tensor(predictions)))
    sess = tf.Session()
    score = round(sess.run(loss) / len(X_test), 4)
    print(score)
    return {'loss': score, 'status': STATUS_OK}

if __name__ == "__main__":
    while True:
        boptimization.run_trials_grid_2(script_name, space, objective)

