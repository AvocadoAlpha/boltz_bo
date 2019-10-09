from keras.datasets import mnist
import numpy as np

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
    x_test = x_test[:2000]
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
    x_test = x_test[:2000]
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
    x_val = x_train[-5000:]
    x_train = x_train[:5000]
    x_test = x_test[:2000]
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
    x_val = x_train[-100:]
    x_train = x_train[:1000]
    x_test = x_test[:1000]
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
    x_val = x_train[-50:]
    x_train = x_train[:500]
    x_test = x_test[:1000]
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
    x_test = x_test[:1000]
    return x_train, x_val, x_test

