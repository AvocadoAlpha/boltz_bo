import sys
import pickle
from hyperopt import space_eval, fmin, Trials, tpe, rand
from keras.datasets import mnist
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
import examples.exhaustive_search
import os

def set_gpu(x):
    if len(x) >= 2:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(x[1])
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def generate_data_huge():
    (x_train,_), (x_test,_) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    noise_factor = 0.0
    x_train = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_train = np.clip(x_train, 0., 1.).astype('float32')
    x_test = np.clip(x_test, 0., 1.).astype('float32')
    x_val = x_train[-1000:]
    x_train = x_train[:49000]
    x_test = x_test[:-1000]
    return x_train, x_val, x_test
def generate_data_big():
    (x_train,_), (x_test,_) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    noise_factor = 0.0
    x_train = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_train = np.clip(x_train, 0., 1.).astype('float32')
    x_test = np.clip(x_test, 0., 1.).astype('float32')
    x_val = x_train[-1000:]
    x_train = x_train[:25000]
    x_test = x_test[:-1000]
    return x_train, x_val, x_test

def generate_data_medium_2():
    (x_train,_), (x_test,_) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    noise_factor = 0.0
    x_train = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_train = np.clip(x_train, 0., 1.).astype('float32')
    x_test = np.clip(x_test, 0., 1.).astype('float32')
    x_val = x_train[-1000:]
    x_train = x_train[:10000]
    x_test = x_test[:2000]
    return x_train, x_val, x_test

def generate_data_medium():
    (x_train,_), (x_test,_) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    noise_factor = 0.0
    x_train = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_train = np.clip(x_train, 0., 1.).astype('float32')
    x_test = np.clip(x_test, 0., 1.).astype('float32')
    x_val = x_train[-1000:]
    x_train = x_train[:5000]
    x_test = x_test[:-1000]
    return x_train, x_val, x_test



def generate_data_small():
    (x_train,_), (x_test,_) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    noise_factor = 0.0
    x_train = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_train = np.clip(x_train, 0., 1.).astype('float32')
    x_test = np.clip(x_test, 0., 1.).astype('float32')
    x_val = x_train[-1000:]
    x_train = x_train[:1000]
    x_test = x_test[:-1000]
    return x_train, x_val, x_test

def generate_data_tiny():
    (x_train,_), (x_test,_) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    noise_factor = 0.0
    x_train = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_train = np.clip(x_train, 0., 1.).astype('float32')
    x_test = np.clip(x_test, 0., 1.).astype('float32')
    x_val = x_train[-1000:]
    x_train = x_train[:500]
    x_test = x_test[:-1000]
    return x_train, x_val, x_test

def generate_data_micro():
    (x_train,_), (x_test,_) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    noise_factor = 0.0
    x_train = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_train = np.clip(x_train, 0., 1.).astype('float32')
    x_test = np.clip(x_test, 0., 1.).astype('float32')
    x_val = x_train[-1000:]
    x_train = x_train[:100]
    x_test = x_test[:-1000]
    return x_train, x_val, x_test

def generate_data_cnn():
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  #
    return x_train, x_val, x_test



def callback(script_name):
    callbacks = [EarlyStopping(monitor='val_loss', patience=15)] #, ModelCheckpoint(filepath='../earlyStops/earlysStop_'+script_name+'_best.h5')]
    return callbacks

def run_trials(script_name, space, objective):

    trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 5  # initial max_trials. put something small to not have to wait

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("../trials/"+script_name, "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=max_trials)

    print("Best:", best)
    print("SpaceEval" + str(space_eval(space, trials.argmin)))

    # save the trials object
    with open("../trials/" + script_name, "wb") as f:
        pickle.dump(trials, f)


def run_trials_grid(script_name, space, objective):

    examples.exhaustive_search.validate_space_exhaustive_search(space)
    trials = Trials()
    best = fmin(objective, space, algo=examples.exhaustive_search.partial(examples.exhaustive_search.suggest, nbMaxSucessiveFailures=1000), trials=trials, max_evals=1000000)

    print("Best:", best)
    print("SpaceEval" + str(space_eval(space, trials.argmin)))

    # save the trials object
    with open("../trials/"+script_name+"_grid", "wb") as f:
        pickle.dump(trials, f)

def run_trials_grid_2(script_name, space, objective):

    examples.exhaustive_search.validate_space_exhaustive_search(space)
    trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 5  # initial max_trials. put something small to not have to wait

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("../trials/"+script_name, "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    best = fmin(objective, space, algo=examples.exhaustive_search.partial(examples.exhaustive_search.suggest, nbMaxSucessiveFailures=1000), trials=trials, max_evals=max_trials)

    print("Best:", best)
    print("SpaceEval" + str(space_eval(space, trials.argmin)))

    # save the trials object
    with open("../trials/" + script_name, "wb") as f:
        pickle.dump(trials, f)

#my_space_eval and pars used for plots (read_trials)






