import sys
import os
sys.path.append(os.path.abspath(os.path.join('..' ,'..', 'src')))
from utils import set_gpu
import utils
set_gpu(sys.argv)
from keras.models import Model
from keras.layers import Dense, Input
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval
from keras.losses import mean_squared_error
from keras import regularizers
import tensorflow as tf
from keras import backend as K
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

# script for training a autoencoder one hidden layer
# this is used to plug in the best values of BO or gridsearch (batch_size and layer1 nodes)
# plots ground-truth, prediction
# plots digits and reconstructed digits

script_name = os.path.basename(__file__).split('.')[0]

x_train, x_val, x_test = utils.generate_data_micro()
#print("medium")
batch_size = 128#int(sys.argv[1]) #1000 #batch_size
layer1 = 300#int(sys.argv[2]) #30 #nodes

script_name = "batch_size="+str(batch_size)+"_units1="+str(layer1)



K.clear_session()


input = Input(shape=(784,))

enc = Dense(layer1, activation='relu')(input)
dec = Dense(784, activation='sigmoid')(enc)
model = Model(input, dec)

encoder = Model(input, enc)

model.compile(loss='mean_squared_error', optimizer='adadelta')
model.fit(x_train, x_train,
                epochs=100,
                batch_size=batch_size,
                shuffle=True,
                validation_data = (x_val, x_val),
                callbacks =utils.callback(script_name))

preds = model.predict(x_test)
loss = tf.keras.backend.sum(mean_squared_error(tf.convert_to_tensor(x_test), tf.convert_to_tensor(preds)))
sess = tf.Session()
loss = sess.run(loss)
score =round(loss/(len(x_test)), 4)
print("Prediction Score :" + str(score))
print(model.summary())


result = model.predict(x_test)

print('done')

x = []
for i in x_test:
    x.extend(i)

y = []
for j in result:
    y.extend(j)


df = pd.DataFrame({"x": x, "y": y})
dfSample = df.sample(30000)  # This is the importante line
xdataSample, ydataSample = dfSample["x"], dfSample["y"]

plot = sns.regplot(x=xdataSample, y=ydataSample, color='red', scatter_kws={'s': 0.1}, line_kws={'color': 'black'})
plot.set_title(script_name+'\n', loc='left')
plot.set_xlabel("ground truth")
plot.set_ylabel("prediction")
plt.grid()
#plt.tight_layout()

plt.savefig('../../plots/prediction-truth/'+str(script_name)+'.png')

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(result[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig('../../plots/prediction-truth/'+str(script_name)+'_digits.png')
plt.show()

