# This program will run multiple S4 simulations and then extract the results# pylint: disable=C0103
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic(u'load_ext autoreload')
from os import system
import subprocess
import numpy as np
import itertools as itt
import multiprocessing
import time
import math
import de_nn as de_nn

from numpy import *
import random
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
#import mxnet.ndarray as nd
#import mxnet as mx
from itertools import chain
import time
import pred as pred



D = 5 
N_P=200
N_I=3
pi=math.pi
t_or_r = 0 # Take 0 for transmittance and 1 for reflectance
L = np.array([25,25,10,10,math.pi/12]) #R1,R2,Phi1,Phi2,Alpha
U = np.array([140,140,100,100,math.pi/6])


def create_population(L,U,N_P,N_I):
    pop = []
    count_isl = 0
    while count_isl < N_I:  
        pop1 = []
        count_pop = 0
        while count_pop < N_P:
            sol = np.zeros(D)
            for i in range(D):
                sol[i] = L[i] + np.random.uniform()*(U[i]-L[i])
            pop1.append(sol)
            count_pop += 1
        pop.append(pop1)
        count_isl += 1
    return(pop)

population = create_population(L,U,N_P,N_I)
# print(population)



red = np.array([0, 0.04095274, 0.12652984, 0.26727547, 0.32790435, 0.3165129 ,
       0.27377142, 0.18395782, 0.09000188, 0.03012615, 0.00461307,
       0.00875541, 0.0595933 , 0.1558087 , 0.27339484, 0.40802109,
       0.55968744, 0.71747317, 0.86264357, 0.96620222, 1.        ,
       0.94389004, 0.80436829, 0.60478253, 0.421672  , 0.26689889,
       0.15524383, 0.08228206, 0.0440595 , 0.02137074, 0.01073244])

green = np.array([0, 0.00120603, 0.0040201 , 0.01165829, 0.02311558, 0.03819095,
       0.06030151, 0.09145729, 0.13969849, 0.20904523, 0.32462312,
       0.50552764, 0.71356784, 0.86633166, 0.95879397, 1.        ,
       1.        , 0.95678392, 0.87437186, 0.76080402, 0.63417085,
       0.50552764, 0.38291457, 0.26633166, 0.1758794 , 0.10753769,
       0.06130653, 0.0321608 , 0.01708543, 0.00824121, 0.0041206])
blue = np.array([0, 1.17036285e-01, 3.64313526e-01, 7.81897184e-01, 9.85892444e-01,
       1.00000000e+00, 9.41933299e-01, 7.26595565e-01, 4.58890582e-01,
       2.62513402e-01, 1.53490209e-01, 8.92726144e-02, 4.41284352e-02,
       2.38135545e-02, 1.14553355e-02, 4.90942949e-03, 2.20077874e-03,
       1.18503470e-03, 9.59313808e-04, 6.20732464e-04, 4.51441792e-04,
       1.69290672e-04, 1.12860448e-04, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00])



target = (1 - blue)*0.7



# %load_ext autoreload
# mse = pred.get_mse(population,target,t_or_r,N_I,N_P)



print("fgsdf", time.time())

de_nn.de_isl(pred.get_mse,target,t_or_r,L,U,num_layers_i = D, num_isles = N_I, num_gens = 1,
	  poplist = population, mut = 0.8, crossp = 0.5, popsize = N_P,
	  its = 1 , lenp = 0.08, lins = 0.06, verbose = 0);

print("fg", time.time())



# This block compares the S4 and NN t or r spectra.
# sol = np.array([97.17444937, 109.60162397,  63.15918391,  21.7148542 ,  0.38959339]) #red/380

# sol = np.array([155.        ,  94.65175338,  51.11965459,  87.59579273, 0.28620164])
sol = np.array([87.34009565, 101.01624446,  15.5869405 ,  41.35939084,
         0.52359878])

get_ipython().system(u'S4 final_geom.lua > 3.txt')
get_ipython().system(u'S4 swgrating.lua > 1.txt')

de_real = np.loadtxt('3.txt')
sur_real = np.loadtxt('1.txt')
sur_pred = pred.predict(np.tile(sol, (10,1)))



fig, ax = plt.subplots(figsize = (10,5), dpi = 300)

plt.plot(de_real[:,0], de_real[:,1]*1.5, color = 'dodgerblue', ls = '--', lw = 3, label = 'DE_S4')
plt.plot(de_real[:,0], sur_real[:,1]*1.5, color = 'blue', lw = 3, label = 'DE_NN (Real)')
plt.plot(de_real[:,0], sur_pred[0, 0:31]*1.5, color = 'k', ls = '--', lw = 1, label = 'DE_NN (Pred)')
plt.plot(de_real[:,0], target*1.25, color = 'k', ls = '-', lw = 1.5, label = 'Target(CMF)')
plt.legend()

plt.xlim(400,700)
plt.ylim(0,0.8)
# plt.box(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()



aa = np.array([0.21175127, 0.18056695, 0.16683779,
       0.16819282, 0.17948126, 0.19382735, 0.20660884, 0.21788057,
       0.23132927, 0.25385392, 0.29379322, 0.3552626 , 0.4283633 ,
       0.50520075, 0.57667125, 0.63455038, 0.67159921, 0.66722777,
       0.61709499, 0.51059013, 0.37504017, 0.23692347, 0.17107751,
       0.14099059, 0.12397095, 0.14686934, 0.20252877, 0.26868318, 0.32398092, 0.47878965,
       0.63338806])

bb = np.array([0.15150344, 0.14012032, 0.14431581,
       0.15798561, 0.17440205, 0.18758564, 0.19439228, 0.19613437,
       0.19751459, 0.2058216 , 0.23138525, 0.28132493, 0.34974268,
       0.42839875, 0.50824103, 0.58105923, 0.64051435, 0.66920001,
       0.66375977, 0.61206964, 0.52539499, 0.37764397, 0.21520157,
       0.09501808, 0.06264498, 0.07538884, 0.13333026, 0.18348681, 0.24267953, 0.36749906,
       0.54808881])

cc  = np.array([0.16953251, 0.15153229, 0.15209587,
       0.15490177, 0.16820861, 0.18205033, 0.19391276, 0.20475367,
       0.21747516, 0.23814297, 0.27399293, 0.33348346, 0.4034388 ,
       0.47764626, 0.5434613 , 0.60388243, 0.64956987, 0.6421439 ,
       0.6045603 , 0.50022256, 0.36679807, 0.23856734, 0.16630831,
       0.1132126 , 0.08946902, 0.13000073, 0.18027249, 0.2334234, 0.299470087, 0.43883094,
       0.6181399 ])

fig, ax = plt.subplots(figsize = (10,5), dpi = 300)

plt.plot(de_real[:,0], aa, color = 'lime', ls = '--', lw = 3, label = 'DE_S4')
plt.plot(de_real[:,0], bb, color = 'green', lw = 3, label = 'DE_NN (Real)')
plt.plot(de_real[:,0], cc, color = 'k', ls = '--', lw = 1, label = 'DE_NN (Pred)')
plt.plot(de_real[:,0], target*1.25, color = 'k', ls = '-', lw = 1.5, label = 'Target(CMF)')
plt.legend()

plt.xlim(400,700)
plt.ylim(0,0.8)
# plt.box(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()


# This block shows the geometry of the unit cell.
# Coemment the print spectra line in lua file (2nd last row)
# Uncomment the print XY line in lua file

get_ipython().system(u'S4 swgrating.lua > 2.txt')
geom = np.loadtxt('2.txt')

plt.plot(geom[0::2], geom[1::2])



get_ipython().system(u'S4 swgrating.lua')



get_ipython().system(u'S4 final_geom.lua')



# xypts = np.loadtxt("blue_rob_xy.dat")
x_srgb = [0.64, 0.30, 0.15, 0.64]
y_srgb = [0.33, 0.60, 0.06, 0.33]

x_fg = [0.27222206478296, 0.34934751403331, 0.45556307617858, 0.27222206478296]
y_fg = [0.26363271688165, 0.51440338179485, 0.38838324248238, 0.26363271688165]

x_sw = [0.2502425414442, 0.36921095580583, 0.45603067743914, 0.2502425414442]
y_sw = [0.25915092291374, 0.51210342897275, 0.37150700845938, 0.25915092291374]

brz_x = [0.29, 0.31, 0.36, 0.29]
brz_y = [0.29, 0.39, 0.32, 0.29]

img = plt.imread("back.png")
fig = plt.figure(figsize=(12,12), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
plt.imshow(img, extent = [0,1,0,1])

ax.plot(x_fg, y_fg, color = 'maroon', ls = '--', label = 'DE_S4')
ax.scatter(x_fg, y_fg, marker = "o" , alpha=1, c='white', edgecolors='black', s=180)

ax.plot(x_sw, y_sw, color = 'blue', ls = '--', label = 'DE_NN')
ax.scatter(x_sw, y_sw, marker = "P" , alpha=1, c='white', edgecolors='black', s=200)

ax.plot(brz_x, brz_y, color = 'k', ls = '--', label = 'BERZINS')
ax.scatter(brz_x, brz_y, marker = "^" , alpha=1, c='white', edgecolors='black', s=200)

ax.plot(x_srgb, y_srgb, color = 'black', ls = '-')
plt.xlim(0,0.8)
plt.ylim(0,0.85)


# plt.xticks([])
# plt.yticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
plt.show()


# xypts = np.loadtxt("blue_rob_xy.dat")
# x_srgb = [0.64, 0.30, 0.15, 0.64]
# y_srgb = [0.33, 0.60, 0.06, 0.33]

x_fg = [0.40302295251611, 0.41309356032056,0.2842817817486,0.40302295251611]
y_fg = [0.44877900383613, 0.27658882559289,0.31677089793388,0.44877900383613]

x_sw = [0.41177284404385, 0.38822005704996,0.2726887103862, 0.41177284404385]
y_sw = [0.40054828714076, 0.27774267888058,0.30694530300841,0.40054828714076]

img = plt.imread("back.png")
fig = plt.figure(figsize=(12,12), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
plt.imshow(img, extent = [0,1,0,1])

ax.plot(x_fg, y_fg, color = 'maroon', ls = '--', label = 'DE_S4')
ax.scatter(x_fg, y_fg, marker = "o" , alpha=1, c='white', edgecolors='black', s=180)

ax.plot(x_sw, y_sw, color = 'blue', ls = '--', label = 'DE_NN')
ax.scatter(x_sw, y_sw, marker = "P" , alpha=1, c='white', edgecolors='black', s=200)

# ax.plot(x_srgb, y_srgb, color = 'black', ls = '-')
plt.xlim(0,0.8)
plt.ylim(0,0.85)


plt.xticks([])
plt.yticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.show()


get_ipython().system(u'S4 swgrating.lua> 4.txt')



geom = np.loadtxt('4.txt')

plt.figure(figsize = (5,5), dpi = 100)
plt.fill(geom[1::2], geom[2::2], color = 'dodgerblue')
plt.plot(geom[1::2], geom[2::2], color = 'k', lw = 0.5)
# plt.scatter(geom[1::2], geom[2::2])
plt.xlim(-geom[0]/2, geom[0]/2)
plt.ylim(-geom[0]/2, geom[0]/2)

plt.xticks([])
plt.yticks([])
plt.show()



get_ipython().system(u'S4 final_geom.lua> 5.txt')



geom = np.loadtxt('5.txt')

plt.figure(figsize = (5,5), dpi = 300)
plt.fill(geom[1::2], geom[2::2], color = 'lightcoral')
plt.plot(geom[1::2], geom[2::2], color = 'k', lw = 0.5)
plt.xlim(-geom[0]/2, geom[0]/2)
plt.ylim(-geom[0]/2, geom[0]/2)

plt.xticks([])
plt.yticks([])
plt.show()



x = loadtxt('xx.txt')
x1 = np.zeros(60)
x2 = np.zeros(60)
x3 = np.zeros(60)
for i in range(60):
    x1[i] = x[i][0]
    x2[i] = x[i][1]
    x3[i] = x[i][2]

    
# x4 = np.linspace(0,60,1)

x11 = np.append(x1[0:20],x2[20:40],axis=0)
x111 = np.append(x11,x3[40:60],axis=0)

x22 = np.append(x2[0:20],x3[20:40],axis=0)
x222 = np.append(x22,x1[40:60],axis=0)

x33 = np.append(x3[0:20],x1[20:40],axis=0)
x333 = np.append(x33,x2[40:60],axis=0)

x4 = np.linspace(1,60,60)
plt.figure(figsize = (5,5), dpi = 300)
plt.plot(x4,x111, lw = 2, color = 'olive')
plt.plot(x4,x222, lw = 2, color = 'royalblue')
plt.plot(x4,x333, lw = 2, color = 'brown')
plt.plot()
plt.xlim(0,60)


x111


x4 = np.linspace(1,60,60)
x4


lay05 = np.array([0.00036534, 0.00024195, 0.00037759, 0.00035435, 0.00028412,
       0.00036514, 0.00034062, 0.00035364, 0.0002644 , 0.0003518 ])

lay10 = np.array([0.0002708 , 0.00015225, 0.00028466, 0.00023427, 0.00024738,
       0.00019605, 0.00023028, 0.00019284, 0.00023497, 0.00019257])

lay15 = np.array([8.75362748e-05, 8.29753657e-05, 8.74895350e-05, 7.96654984e-05,
       8.74049254e-05, 8.39147150e-05, 8.32967492e-05, 7.23273881e-05,
       7.18733319e-05, 7.04023204e-05])

lay20 = np.array([4.69863729e-05, 4.41485809e-05, 3.98375667e-05, 4.17357079e-05,
       4.21901665e-05, 3.91243490e-05, 3.63351982e-05, 4.40871502e-05,
       4.58873966e-05, 4.24468623e-05])

lay25 = np.array([5.23394222e-05, 4.51676912e-05, 4.66001081e-05, 5.87733355e-05,
       4.21651926e-05, 5.53585106e-05, 4.41639651e-05, 5.29742713e-05,
       4.79378483e-05, 4.25400830e-05])

lay30 = np.array([4.60785656e-05, 4.45131451e-05, 5.14885341e-05, 5.30751176e-05,
       5.00589932e-05, 4.22049152e-05, 4.11478446e-05, 5.98100984e-05,
       5.93002374e-05, 5.82048245e-05])



layers = (lay05, lay10, lay15, lay20, lay25, lay30)
fig1, ax1 = plt.subplots(figsize = (5,5), dpi = 600)
box=plt.boxplot(layers, notch=False, whis=2.5,showfliers=False,
                widths=(0.7,0.7,0.7,0.7,0.7,0.7),
                patch_artist=True, vert=True,labels=('','','','','',''))
plt.box(None)



colors = ['cornflowerblue','cyan','lightseagreen','lightgreen','pink','maroon']
hatches = ['//','//','//','//','//','//']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    
for patch, hatchh in zip(box['boxes'], hatches):
    patch.set(hatch=hatchh)
    
plt.grid(axis='y')
# plt.xticks([1.5,4.5,7.5,10.5,13.5],['400k','320k','160k','80k','40k'],fontsize=12)
plt.yticks([0.00003,0.00006,0.00009,0.00012,0.00015,0.00018])
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
plt.yscale('log')
plt.ylim(0.00001, 0.0010)
plt.show()



ds006 = np.array([0.00439904, 0.00469207, 0.00550741, 0.004024  , 0.00431153,
       0.00457451, 0.00597811, 0.00420901, 0.00557594, 0.00457047])

ds012 = np.array([0.00098072, 0.00086926, 0.00082669, 0.00093198, 0.00097575,
       0.00085804, 0.00081431, 0.00093871, 0.00092194, 0.00089017])

ds025 = np.array([0.00026512, 0.00037752, 0.00037077, 0.00032883, 0.00024012,
       0.00034818, 0.00034118, 0.00034917, 0.00031684, 0.00035023])

ds050 = np.array([8.73292974e-05, 7.24885728e-05, 8.39423185e-05, 7.99450559e-05,
       8.19370101e-05, 7.93321673e-05, 8.61862022e-05, 8.50611002e-05,
       7.92311060e-05, 8.43442288e-05])

ds100 = np.array([4.27337741e-05, 4.16296919e-05, 4.03432928e-05, 3.86369103e-05,
       3.46476946e-05, 4.97654888e-05, 4.07742700e-05, 3.18428639e-05,
       3.29416802e-05, 3.26309735e-05])

ds200 = np.array([2.94536530e-05, 3.41341335e-05, 3.13281631e-05, 2.83073228e-05,
       3.40562281e-05, 4.43505052e-05, 3.29488180e-05, 4.25627184e-05,
       3.09845935e-05, 3.72908075e-05])



dsets = (ds006, ds012, ds025, ds050, ds100, ds200)
fig1, ax1 = plt.subplots(figsize = (5,5), dpi = 600)
box=plt.boxplot(dsets, notch=False, whis=2.5,showfliers=False,
                widths=(0.7,0.7,0.7,0.7,0.7,0.7),
                patch_artist=True, vert=True,labels=('','','','','',''))
plt.box(None)



colors = ['cornflowerblue','cyan','lightseagreen','lightgreen','pink','maroon']
hatches = ['//','//','//','//','//','//']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    
for patch, hatchh in zip(box['boxes'], hatches):
    patch.set(hatch=hatchh)
    
plt.grid(axis='y')
# plt.xticks([1.5,4.5,7.5,10.5,13.5],['400k','320k','160k','80k','40k'],fontsize=12)
plt.yticks([0.00003,0.00006,0.00009,0.00012,0.00015,0.00018])
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
plt.yscale('log')
plt.ylim(0.00001, 0.010)
plt.show()



# temp = np.zeros(10)
# for i in range (10):
#     x = (2.5 + 2 * np.random.random()) * (10**(-5))
#     print("{:e}".format(x))
#     temp[i] = x
# temp

