

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
import math



pi = math.pi
xx = np.loadtxt('100k_str.txt')
lower = np.array([25,25,10,10,pi/12]) #R1,R2,Phi1,Phi2,Alpha
upper = np.array([155,155,100,100,pi/6])
for i in range (np.shape(xx)[0]):
    xx[i] = (xx[i]-lower)/(upper-lower)
yy = np.loadtxt('all_spectra.dat')
yy1 = np.reshape(yy, (100000,31,2))



yy2 = np.swapaxes(yy1,1,2)
yy3 = np.reshape(yy2, (100000,62))



plt.plot(yy[0:31,1])
plt.plot(yy3[0,31:62])



get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import os
np.set_printoptions(threshold=1000000)

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Activation, BatchNormalization
from keras.layers import Dropout, UpSampling2D, ZeroPadding2D, AveragePooling2D, Conv2DTranspose
from keras.layers import Flatten, Cropping1D, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.engine.topology import Layer
from keras.layers import Lambda, Input
import tensorflow as tf
from itertools import product



x_tr,x_te,y_tr,y_te= train_test_split(xx, yy3, test_size=0.3)
print(np.shape(x_tr), np.shape(x_te), np.shape(y_tr), np.shape(y_te))



def std_dev(actu, pred):
    return keras.backend.std(keras.backend.abs(actu - pred))

def mmae(actu, pred):
    return keras.backend.max(keras.backend.abs(actu - pred))



# Statistical analysis

activ = np.array(["sigmoid","relu","tanh"])
lossfn = np.array(['mse'])


    
for i in range(3):
    
    with open ('error_mse.txt', 'a') as file1:
        file1.write(str(activ[i]) + '\n')
        
        for j in range(1):
            for k in range(5):

                iput = Input(shape=(5,))
                dens1 = Dense(256, activation="relu")(iput)
                dens2 = Dense(256, activation="relu")(dens1)
                dens3 = Dense(256, activation="relu")(dens2)
                dens4 = Dense(256, activation="relu")(dens3)

                dens5 = Dense(256, activation="relu")(dens4)
                dens6 = Dense(256, activation="relu")(dens5)
                dens7 = Dense(256, activation="relu")(dens6)
                dens8 = Dense(256, activation="relu")(dens7)

                dens9 = Dense(256, activation="relu")(dens8)
                dens10 = Dense(256, activation="relu")(dens9)
                dens11 = Dense(256, activation="relu")(dens10)
                dens12 = Dense(256, activation="relu")(dens11)

                dens13 = Dense(256, activation="relu")(dens12)
                dens14 = Dense(256, activation="relu")(dens13)
                dens15 = Dense(256, activation="relu")(dens14)
                dens16 = Dense(256, activation="relu")(dens15)

                dens17 = Dense(256, activation="relu")(dens16)
                dens18 = Dense(256, activation="relu")(dens17)
                dens19 = Dense(256, activation="relu")(dens18)
                dens20 = Dense(256, activation="relu")(dens19)



                densf = Dense(62, activation=activ[i])(dens20)
                # oput = Activation('sigmoid')(densf)

                model1 = Model(inputs = iput, outputs = densf)
                model1.summary()

                model1.compile(optimizer = 'adam',
                               loss = lossfn[j],
                               metrics=[mmae, 'mse'])
                hstry3 = model1.fit(x_tr, y_tr, epochs = 50,
                                    batch_size=256, shuffle=True,
                                    validation_data=(x_te, y_te))


                y_pre = model1.predict(x_te)
                err = np.mean((y_te - y_pre)**2)
                # print(err)
                file1.write(str(err)+'\t')

            file1.write('\n')



iput = Input(shape=(5,))
dens1 = Dense(256, activation="relu")(iput)
dens2 = Dense(256, activation="relu")(dens1)
dens3 = Dense(256, activation="relu")(dens2)
dens4 = Dense(256, activation="relu")(dens3)

dens5 = Dense(256, activation="relu")(dens4)
dens6 = Dense(256, activation="relu")(dens5)
dens7 = Dense(256, activation="relu")(dens6)
dens8 = Dense(256, activation="relu")(dens7)

dens9 = Dense(256, activation="relu")(dens8)
dens10 = Dense(256, activation="relu")(dens9)
dens11 = Dense(256, activation="relu")(dens10)
dens12 = Dense(256, activation="relu")(dens11)

dens13 = Dense(256, activation="relu")(dens12)
dens14 = Dense(256, activation="relu")(dens13)
dens15 = Dense(256, activation="relu")(dens14)
dens16 = Dense(256, activation="relu")(dens15)

dens17 = Dense(256, activation="relu")(dens16)
dens18 = Dense(256, activation="relu")(dens17)
dens19 = Dense(256, activation="relu")(dens18)
dens20 = Dense(256, activation="relu")(dens19)



densf = Dense(62, activation='sigmoid')(dens20)
# oput = Activation('sigmoid')(densf)

model1 = Model(inputs = iput, outputs = densf)
model1.summary()

model1.compile(optimizer = 'adam',
               loss = 'mse',
               metrics=[mmae, 'mse'])
hstry3 = model1.fit(x_tr, y_tr, epochs = 50,
                    batch_size=256, shuffle=True,
                    validation_data=(x_te, y_te))


y_pre = model1.predict(x_te)
err = np.mean((y_te - y_pre)**2)


np.argmax(y_te - y_pre)/62


model1.save_weights('20layer_256neu.h5')



err



rnd = np.random.randint(0, np.shape(y_pre)[0])
rnd = 19106
# fig = plt.figure()
fig, plt1 = plt.subplots(1,2, figsize=(15,5),dpi=300)
plt1[0].plot(y_te[rnd][0:31], label='True value',linestyle='dashed',color='black', lw = 3)
plt1[0].plot(y_pre[rnd][0:31],label='Predicted value',color='brown', lw = 3)
plt1[0].set_xlim(-1,31)
plt1[0].set_ylim(0,1)
plt1[0].set_xlabel('Wavelength (nm)', fontsize = 12)
plt1[0].set_ylabel('Transmittance', fontsize = 12)
# plt1[0].rcParams.update({'figure.figsize':(7,5), 'figure.dpi':300})
plt1[0].legend(fontsize = 12)

plt1[1].plot(y_te[rnd][31:62], label='True value',linestyle='dashed',color='black', lw = 3)
plt1[1].plot(y_pre[rnd][31:62],label='Predicted value',color='brown', lw = 3)
plt1[1].set_xlim(-1,31)
plt1[1].set_ylim(0,1)
plt1[1].set_xlabel('Wavelength (nm)', fontsize = 12)
plt1[1].set_ylabel('Reflectance', fontsize = 12)
# plt1[1].rcParams.update({'figure.figsize':(7,5), 'figure.dpi':300})
plt1[1].legend(fontsize = 12)

plt.show()



xx1 = np.reshape(xx, (500000,1))
plt.figure(figsize = (6,5), dpi = 300)
plt.hist(xx1, bins = 50, normed = 1, color = 'sandybrown')
plt.show()

