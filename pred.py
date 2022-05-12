# %load_ext autoreload
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
import math

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

from keras.layers import Input, Dense, Add
from keras.models import Model
from itertools import chain


import random
from datetime import datetime

# red = np.array([0.04095274, 0.12652984, 0.26727547, 0.32790435, 0.3165129 ,
#        0.27377142, 0.18395782, 0.09000188, 0.03012615, 0.00461307,
#        0.00875541, 0.0595933 , 0.1558087 , 0.27339484, 0.40802109,
#        0.55968744, 0.71747317, 0.86264357, 0.96620222, 1.        ,
#        0.94389004, 0.80436829, 0.60478253, 0.421672  , 0.26689889,
#        0.15524383, 0.08228206, 0.0440595 , 0.02137074, 0.01073244, 0.00546037])


# target = red
# order = 0
# pol = 0

def predict(all_structures):
    structures = np.copy(all_structures)
    lower = np.array([25,25,10,10,math.pi/12]) #R1,R2,Phi1,Phi2,Alpha
    upper = np.array([140,140,100,100,math.pi/6])
    for i in range (np.shape(structures)[0]):
        structures[i] = (structures[i]-lower)/(upper-lower)
     
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

    model1 = Model(inputs = iput, outputs = densf)
    model1.load_weights('330_20layer_256neu.h5')
    
    return model1.predict(structures)




def get_mse(population,target,t_or_r,N_I,N_P):
    target = np.tile(target,(N_I*N_P,1))
    isl = list(chain.from_iterable(population))
#     print(np.shape(isl))
    prediction = predict(isl)
    prediction = prediction[:,t_or_r*31:t_or_r*31+31]
#     print(np.shape(prediction))
    m = np.mean((prediction-target)**2, axis = 1)
#     print(m)
    return np.reshape(m,(N_I,N_P))



