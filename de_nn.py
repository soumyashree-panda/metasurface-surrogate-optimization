import numpy as np
import random
from itertools import chain
import pred as pred

def lengthen_best(best):
    return np.append(best, np.random.rand())

def ins_best(best):
    ins_loc = np.random.choice(best.size)
    return np.insert(best, (ins_loc, ins_loc), np.random.uniform(0,1,2))


def cre_mutant(best, b, c, mut, L, U):
    mut_V = np.zeros(len(best))
    for i in range(len(best)):
        mut_V[i] = np.clip(best[i] + mut*(b[i] - c[i]),L[i],U[i] )
    return mut_V


def crossov(mutant, curr, crossp):
    cross_points = np.random.rand(curr.size) > crossp
    return np.where(cross_points, mutant, curr)


def de_isl(
    fobj,# objective function, prediction
    target,
    t_or_r,
    L,
    U,
    num_layers_i=16,
    num_isles = 5, #number of islands
    num_gens =5,
    poplist=[],  # already initialized population
    mut=0.8,  # mutation rate
    crossp=0.7,  # crossover
    popsize=20,  # the population size
    its=1000, # the number of iterations needed to run for
    lenp=0.08,
    lins=0.06,
    verbose=0):

    """
        This function performs diff evolution using fobj evaluated in parallel  
        returns the  last pop, fitness of pop, best individual and opt history 
    """
    history = []
    bids = np.zeros(num_isles, dtype=int)
    bfits = np.zeros(num_isles)
    mut_r = np.random.uniform(0.4, 0.8, num_isles)
    cro_r = np.random.uniform(0.5, 1.0, num_isles)
    num_func_evals = 0
  
    trilist = []
    for isl in range(num_isles):
        cc=[]
        for j in range(popsize):
            sol1=np.zeros(num_layers_i)
            for k in range(num_layers_i):
                sol1[k] = L[k]+np.random.random()*(U[k]-L[k])
            cc.append(sol1)
        trilist.append(cc)

        
    print(np.shape(trilist))    
    tmp2 = np.random.uniform(0,1, num_isles*num_layers_i)
    bests = np.split(tmp2, num_isles)
#     print("bests=   ",bests)
    

    for gen in range(num_gens):
#         print(history)
        if verbose == 0:
            print("==============================")
            print("Epoch #:" + str(gen + 1))
            print("==============================")
#         with open("Data_best.txt","a") as data:
# #             data.write("Generation : "+ str(gen))
#         data.close()
#         with open("Data_bfits.txt","a") as data1:
# #             data1.write("Generation : "+str(gen))
#         data1.close()        
        isl = list(chain.from_iterable(poplist))
#         print("\n isl=",isl)
        
        fitness = fobj(poplist,target,t_or_r,num_isles,popsize)
        num_func_evals+=len(isl)
#         fitness=np.transpose(fitness)
        print(np.shape(fitness))
        
        for isln in range(num_isles):
            bids[isln] = np.argmin(fitness[isln])
            bests[isln] = poplist[isln][bids[isln]]

#         print("\n bids bests",bids,bests)  
        
        for i in range(its):
            for isln in range(num_isles):
                j = 0
                while (j < popsize):
                    idxs = [idx for idx in range(popsize) if idx != j]
                    picks = np.random.choice(idxs, 3, replace = False)
#                     print(j,picks)
                    a, b, c = poplist[isln][picks[0]],  poplist[isln][picks[1]], poplist[isln][picks[2]]
                    mutant = cre_mutant(a, b, c, mut_r[isln],L,U)
                    child = crossov(mutant, poplist[isln][j], cro_r[isln])
                    trilist[isln][j] = child
                    j += 1

            tflat = list(chain.from_iterable(trilist))
            f = fobj(trilist,target,t_or_r,num_isles,popsize)
#             f = np.transpose(f)
#             print("\n asdfg",f_isl)

            # print("F_isl =  ",f_isl) 

#             f_isl_gpu = f_isl

#             f = np.split(f_isl, num_isles)
            num_func_evals+=len(tflat)
            
            #print("***********TRILIST***************") 
            #print(trilist)
            #print("***********POPLIST***************")
            #print(poplist)
#            print("Trilist Fitness = " ,f,"\n","Population Fitness =",fitness)
            for isln in range(num_isles):
                for j in range(popsize):
#                     print(isln,j,f[isln][j],fitness[isln][j])
                    if (f[isln][j] < fitness[isln][j]):
                        fitness[isln][j] = f[isln][j]
                        poplist[isln][j] = trilist[isln][j]
                        if f[isln][j] < fitness[isln][bids[isln]]:
                            bids[isln] = j
                            bests[isln] = trilist[isln][j]

                bfits[isln] = fitness[isln][bids[isln]]
                

            #print("**********UPDATED POPLIST***********\n",poplist)
            if (i+1)%1 == 0:
                print("Iteration = ","%3d" %i,  " -- ")
#                 for num in bfits:
#                     print("Best fit = ","%4.3f" %num, "  ")
#                 print()
            with open("Data_bfits.txt","a") as data:
                data.write(str(i)+"\t"+str(bfits)+"\n")
            data.close()
            
            
            with open("Data_best.txt","a") as data1:
                np.savetxt(data1,bests,fmt='%4.3f')
            data1.close()
            
            print('=====',bfits, '======')
            history.append(np.copy(bfits))



        if its > 64:
            its = int(its/2)



        if gen < (num_gens - 1):
            #print("Round robin best migration")
            stmp = np.copy(poplist[num_isles-1][bids[num_isles-1]])
            for isln in range(num_isles-1, 0, -1):

                poplist[isln][bids[isln]] = np.copy(poplist[isln-1][bids[isln-1]])
            poplist[0][bids[0]] = stmp 


    print("Num func evals: ", num_func_evals)
    print("bid = ",bids,"bests = ", bests,"bfits = ", bfits,"history = ", np.asarray(history))
    return bids, bests, bfits, np.asarray(history), num_func_evals





