# coding: utf-8
# This code creates multiple folders, divides the polygon shapes and move into the folders; Determines the spectra of structures from all folders using S4 parallely; Writes the transmittance and reflectance.
import numpy as np
from itertools import product
import subprocess
import os
x_str=np.loadtxt('100k_str330.txt')
# Read the polygon structures from the file. Divide them into multiple files.
count = 0
line_no = 0
structure_per_sim = 1000   # Each file will contain this number of polygons. Names of files will be 0.txt, 1.txt and so on.
with open("100k_str330.txt") as f1:
    count = 0
    for line in f1:
        if (line_no%structure_per_sim == 0):
            filename = str(count)+".txt"
            count = count+1
        line_no = line_no+1
        with open(filename,'a') as f2:
            f2.write(line)
path = "/dset330/"  #Replace this given path with the path to your folder.
# This block of codes creates folders with name '0', '1', '2' and so on; copies '0.txt' to '/0' and so on and rename them as 'structures.txt'
# swrating.lua is also copied. Remember to change the periodicity in swgrating.lua.
for i in range (count):
    mkdir = "mkdir temp" + str(i)
    copy_structures = "cp "+str(i)+".txt ""temp"+str(i)+"/"
    rename_structures = "mv ""temp"+str(i)+"/"+str(i)+".txt ""temp"+str(i)+"/"+"structures.txt"
    copy_lua = "cp ""swgrating.lua ""temp"+str(i)+"/"
    os.system(mkdir)
    os.system(copy_structures)
    os.system(copy_lua)
    os.system(rename_structures)
# This block of codes runs S4 on all the polygon shapes.
#Go in to each folder and run 'swgrating.lua'. 
# 'swgrating.lua' reads the geometries from 'structures.txt' in its respective folder and gives their coresponding spectrum in 'spectra.dat'
for i in range (count):
    os.chdir(path+"temp"+str(i)+"/")
    run_s4 = "S4 swgrating.lua"
    subprocess.Popen(run_s4, shell = True)
path = "/dset330/"  #Replace the given path with the path to your folder.
# This block of codes retrieve 'spectra.dat' from each folder and write them to a file called 'all_spectra.dat'
for i in range (count):
    os.chdir(path+"temp"+str(i)+"/")
    out = np.loadtxt('spectra.dat')
    os.chdir(path)
    with open ("all_spectra.dat","a") as f:
        for j in out:            
            for k in j:
                f.write(str(k)+'\t')
            f.write('\n')
# In case you need to delete the files and folders
# for i in range (count):
#     delete = "rm " + str(i)+".txt"
#     os.system(delete)
    
# for i in range (count):
#     delete = "rm -rf temp" + str(i)
#     os.system(delete)

