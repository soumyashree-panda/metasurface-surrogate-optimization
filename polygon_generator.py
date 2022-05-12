# This file generates random polygons and write them to '330_polygons.txt'

import numpy as np
import math
import matplotlib.pyplot as plt

pi = math.pi
lower = np.array([25,25,10,10,pi/12]) #R1,R2,Phi1,Phi2,Alpha
upper = np.array([155,155,100,100,pi/6])
R1 = np.random.uniform(25,140,100000)
R2 = np.random.uniform(25,140,100000)
Phi1 = np.random.uniform(10,100,100000)
Phi2 = np.random.uniform(10,100,100000)
Alpha = np.random.uniform(pi/12,pi/6,100000)
strct = []
for i in range(100000):
    xx = np.zeros(5)
    for j in range (5):
        xx[j] = lower[j] + np.random.uniform(0,1)*(upper[j] - lower[j])
    print(xx)
    strct.append(xx)
np.savetxt('330_polygons.txt',strct, fmt='%5.5f', delimiter = '\t')

