#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1

import random
import re
import subprocess
import getcycle
import getpgm
import os

from env import Env
from  subprocess import call
from multiprocessing.dummy import Pool as ThreadPool

from deap import base
from deap import creator
from deap import tools
import numpy as np
import time 

def geo_mean(iterable):
    a = np.array(iterable).astype(float)
    prod = a.prod()
    prod = -prod if prod < 0 else prod
    return prod**(1.0/len(a))

NUM_PASSES = 45
def setupGA(envs, length):
    # Weights=1 maximize a single objective
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator
    #                      define 'attr_bool' to be an attribute ('gene')
    #                      which corresponds to integers sampled uniformly
    #                      from the range [0,1] (i.e. 0 or 1 with equal
    #                      probability)

    #toolbox.register("attr_bool", random.randint, 0, getcycle.countPasses()-2)
    toolbox.register("attr_bool", random.randint, 0, NUM_PASSES)

    # Structure initializers
    #                         define 'individual' to be an individual
    #                         consisting of 100 'attr_bool' elements ('genes')
    # Number of optimization passes applied
    toolbox.register("individual", tools.initRepeat, creator.Individual,
        toolbox.attr_bool, length)

    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # the goal ('fitness') function to be maximized
    # Run Legup and recorde the negative cycles

    def evalOneMax(individual):
      #_, reward =env.reset(init=individual, get_obs=False)
      #pool = ThreadPool(len(envs))
      rews = [env.reset(init=individual, get_obs=False)[1] for env in envs]
      #rews = pool.map(lambda env: env.reset(init=individual, get_obs=False)[1], envs)
      reward = -geo_mean(rews)
      
      #print(reward)
      #if (-reward < 6857):
      #  print("{}|{}\n".format(-reward, individual))

      return reward,

    #----------
    # Operator registration
    #----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalOneMax)

    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # Pick top 3
    toolbox.register("select", tools.selTournament, tournsize=5)
    return toolbox
    #----------

def trainGA(toolbox):
    random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=300)
    #pop = toolbox.population(n=2)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    #CXPB, MUTPB = 0.5, 0.2
    CXPB, MUTPB = 0.6, 0.3

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    # If the best gnome is larger than 0?
    #while max(fits) < 0 and g < 1:
    
    begin = time.time()
    while max(fits) < 0 and g < 30:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        end = time.time()
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        compile_time = end - begin
        print("  Time {}".format(int(compile_time)))
        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        print("Best is {}, {}".format(getcycle.getPasses(best_ind), best_ind.fitness.values))

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    print("Best is {}, {}".format(getcycle.getPasses(best_ind), best_ind.fitness.values))
    return best_ind.fitness.values, best_ind

def ga_test(length, test_name):
    #pgms = getpgm.lsFiles("../dataset")

    #from chstone_bm import get_chstone, get_others
    #bm = get_chstone()[4:5]
    #bm = [("gsm.c", "/scratch/qijing.huang/LegUp/legup-4.0/examples/chstone/gsm/") ]
    #print(len(bm))
    #bm.extend(get_others())

    #test_len = [6, 12, 24, 48, 96]
    #test_len = [48, 96]
    #test_len = [5]

    fout = open("report_ga"+".txt", "w")
    fout.write("Benchmark |Cycle Counts | Algorithm Runtime (s)|Passes \n")

    #for pgm, path in bm:
    for pgm_name in ["greedy"]:
        from chstone_bm import get_bms
        #bm = get_chstone(N=4)
        bm = get_bms(test_name) 
        envs = []
        i = 0
        for pgm, path in bm:
            envs.append(Env(pgm, path, "run_ga_"+str(i)))
            i = i+1

        #env=Env(pgm, path, init_with_passes=True)  
        print("Program: %s" % pgm_name)
        toolbox = setupGA(envs, length=length)
        begin = time.time()
        cycles, passes = trainGA(toolbox)
        end = time.time()
        compile_time = end - begin
        print("Compile Time: %d"%(int(compile_time)))
        fout.write("{}|{}|{}|{}\n".format(pgm, cycles[0], compile_time, passes))

                 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', '-bm', type=str, default=" ")
    parser.add_argument('--length', '-len', type=int, default=3)
    args = parser.parse_args()
    ga_test(args.length, args.benchmark)
