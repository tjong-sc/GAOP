from GAOP.Representations.QASMPrimitiveInstruction import QASMPrimitiveInstruction
from GAOP.QiskitDrivers.OpenQASMConverter import OpenQASMConverter
from GAOP.QiskitDrivers.OpenQASMAdapter import OpenQASMAdapter
from GAOP.GeneticAlgorithms.Population import Population
from GAOP.GeneticAlgorithms.Ga import Ga
from GAOP.Statistics.GAEpochStat import GAEpochStat
from GAOP.Statistics.LineGraph import LineGraph
from GAOP.Statistics.RandomWrapper import RandomWrapper
from GAOP.GeneticAlgorithms.Chromosome import Chromosome
from GAOP.GeneticAlgorithms.GAParameters import GAParameters
from incdeclib.IncDecExperiment import IncDecExperiment
from random import seed, random
from typing import List, Tuple
from math import pi, ceil
import traceback
import random
import itertools

#global parameters that control the GA
set = "A"
if(set == "A"):
    CProb = 0.65
    MProb = 0.001
else:
    CProb = 0.85
    MProb = 0.01

#5 qubit increment & dec problems
registers = 5
totalEpochs = 20
popSize = 30
minSize = 5
maxSize = 25
mStartEpoch = 5
mEndEpoch = 8
cStartEpoch = 3
cEndEpoch = 8
eliteRatio = 0.1


#if you want a random seed, then do the following
r = RandomWrapper()
#r = RandomWrapper(222)

def main():
    # this is the "prepare" statements used by the maxone problem experiments
    hadamardPattern = ""
    # places all gates in hadamard
    for i in range(0, registers):
        hadamardPattern = hadamardPattern + "h q[" + str(i) + "];\n"

    #this is the first experiment:
    #Roulette wheel selection,
    #default multipoint crossover, mutation, and selection function is based on
    #the binary string problem and how close the value is to the intended 1111 string value.
    #1/(max value - prob of the largest binary string in the quantum experiment) where max value is 2^ num of register - 1 (the max value
    #for a binary string represented by the num of registers).
    params = GAParameters(MProb, CProb, totalEpochs, openQASMPrepareStatement = hadamardPattern)
    try:
        ga = Ga(r, popSize, registers, generationFunction = QASMPrimitiveInstruction, minSize = minSize, maxSize = maxSize)
        stats = ga.evolve(totalEpochs, probCrossover = CProb, probMutation= MProb, params = params)
    except Exception:
        traceback.print_exc()
    generateSummaries(stats = stats, experimentName="none")
    
    #this is the 2nd experiment:
    #Roulette wheel selection,
    #default multipoint crossover, mutation, selection function is, again, based on the binary string
    #problem, however, the fitness value is weighted based on the following formulation:
    #prob1 * value of the first binary string + prob2 * value of the 2nd binary string + ...    
    params = GAParameters(MProb, CProb, totalEpochs, openQASMPrepareStatement = hadamardPattern)

    try:
        ga = Ga(r, popSize, registers, generationFunction = QASMPrimitiveInstruction, minSize = minSize, maxSize = maxSize)        
        stats = ga.evolve(totalEpochs, selectionFunc = weightedSelectionFunc, probCrossover = CProb, probMutation= MProb, params = params)
    except Exception:
        traceback.print_exc()
    generateSummaries(stats = stats, experimentName="W")
    
    #This is the start of the 3rd experiment with decreasing M and C probabilities
    #Default roulette selection method
    params = GAParameters(MProb, CProb, totalEpochs, decreaseM = True, decreaseC = True, \
                MStartEpoch = mStartEpoch, MEndEpoch = mEndEpoch, CStartEpoch = cStartEpoch, CEndEpoch = cEndEpoch, \
                openQASMPrepareStatement = hadamardPattern)
    try:
        ga = Ga(r, popSize, registers, generationFunction = QASMPrimitiveInstruction, minSize = minSize, maxSize = maxSize)
        stats = ga.evolve(totalEpochs, probCrossover = CProb, probMutation= MProb, params = params)
    except Exception:
        traceback.print_exc()
    generateSummaries(stats = stats, experimentName="dMC")

    #This is the start of the 4th experiment with decreasing M and C probabilities
    #weight average selection method    
    params = GAParameters(MProb, CProb, totalEpochs, decreaseM = True, decreaseC = True, \
                MStartEpoch = mStartEpoch, MEndEpoch = mEndEpoch, CStartEpoch = cStartEpoch, CEndEpoch = cEndEpoch, \
                openQASMPrepareStatement = hadamardPattern)
    try:
        ga = Ga(r, popSize, registers, generationFunction = QASMPrimitiveInstruction, minSize = minSize, maxSize = maxSize)
        stats = ga.evolve(totalEpochs, selectionFunc = weightedSelectionFunc, probCrossover = CProb, probMutation= MProb, params = params)
    except Exception:
        traceback.print_exc()
    generateSummaries(stats = stats, experimentName="dMCW")
    
    #This is the start of the 5th experiment with decreasing M and C probabilities
    #weight average selection method and delta mutation func
    
    params = GAParameters(MProb, CProb, totalEpochs, decreaseM = True, decreaseC = True, \
                MStartEpoch = mStartEpoch, MEndEpoch = mEndEpoch, CStartEpoch = cStartEpoch, CEndEpoch = cEndEpoch, \
                openQASMPrepareStatement = hadamardPattern)
    try:
        ga = Ga(r, popSize, registers, generationFunction = QASMPrimitiveInstruction, minSize = minSize, maxSize = maxSize)
        stats = ga.evolve(totalEpochs, selectionFunc = weightedSelectionFunc, probCrossover = CProb, \
            probMutation= MProb, mutationArgFunc = deltaMutationFunc, params = params)
    except Exception:
        traceback.print_exc()
    generateSummaries(stats = stats, experimentName="dMCW")
    
    #test code for elitism
    params = GAParameters(MProb, CProb, totalEpochs, decreaseM = True, decreaseC = True, \
                MStartEpoch = mStartEpoch, MEndEpoch = mEndEpoch, CStartEpoch = cStartEpoch, CEndEpoch = cEndEpoch, \
                eliteRatio=eliteRatio , openQASMPrepareStatement = hadamardPattern)
    try:
        ga = Ga(r, popSize, registers, generationFunction = QASMPrimitiveInstruction, minSize = minSize, maxSize = maxSize)
        stats = ga.evolve(totalEpochs, selectionFunc = elitismDefaultSelectionFunc, probCrossover = CProb, \
            probMutation= MProb, mutationArgFunc = deltaMutationFunc, params = params)
    except Exception:
        traceback.print_exc()
    generateSummaries(stats = stats, experimentName="edMCW")
    input("Press [enter] to continue.")

def incrementExperiment():
    """ 
    This is the main method for the increment problem 
    This problem is phrased as follows:
    given a set of registers, we will create a prepare statement corresponding to a value in superposition, such as 
    4, the increment op should output 50% for 1 and 5 correspondingly
    
    The setup is the same as what's depicted in : https://oreilly-qc.github.io/# (increment & decrement)
    but we will use a[5] instead of a[4] and scratch[1] in the preparestatement. Thus, wherever there's a scratch[0], 
    we will have a[4] instead. If you look at the test code in OpenQASMTest.py, the output should be the same.

    We will implement the experiment for 5 qubits (+1 qubit as scratch): 
    the first qubit is the min value within the circuit: x a[0] 
    (as in the prepareStatement)
    So the values in the lookup table are : 
    h a[1] (2^1) - {'00010': 0.5, '00100': 0.5}
    h a[2] (2^2) - {'00010': 0.5, '00110': 0.5}
    h a[3] (2^3) - {'00010': 0.5, '01010': 0.5}
    h a[4] (2^4) - {'00010': 0.5, '10010': 0.5}
    Note that the lowest value is 00010 = 2 because we wrote a value of 1 in x q[0].

    """
    #sets up entries in the lookup table
    lookupTable = [
        ('00010','00100', "x q[0];\nh q[1];\nrz(0.785398163397448) q[1];\n"),
        ('00010','00110', "x q[0];\nh q[2];\nrz(0.785398163397448) q[2];\n"),
        ('00010','01010', "x q[0];\nh q[3];\nrz(0.785398163397448) q[3];\n"),
        ('00010','10010', "x q[0];\nh q[4];\nrz(0.785398163397448) q[4];\n"),
    ]
    
    #this array makes it easy to control which experiment is to run by the sim
    enabledExperiments = [False,False,False,True]
    #default params
    params = GAParameters(MProb, CProb, totalEpochs)

    incDecExp = IncDecExperiment(registers, lookupTable, r, popSize)

    if(enabledExperiments[0]):
        # the default roulette wheel selection with a constant CProb, MProb, 80% retention rate            
        try:
            ga = Ga(r, popSize, registers, generationFunction = QASMPrimitiveInstruction, minSize = minSize, maxSize = maxSize)
            stats = ga.evolve(totalEpochs, probCrossover = CProb, probMutation= MProb, \
                        evaluationFunc = incDecExp.evaluationFunction, selectionFunc= incDecExp.rouletteWheelSelection, \
                        params = params)
        except Exception:
            traceback.print_exc()
        generateSummaries(stats = stats, experimentName="inc-none")
    
    if(enabledExperiments[1]):
        
        try:
            # decreasing CProb, MProb with default roulette wheel selection

            # params = GAParameters(MProb, CProb, totalEpochs, decreaseM = True, decreaseC = True, \
            #            MStartEpoch = mStartEpoch, MEndEpoch = mEndEpoch, CStartEpoch = cStartEpoch, CEndEpoch = cEndEpoch)

            #constant CProb, MProb with default roulette wheel selection
            params = GAParameters(MProb, CProb, totalEpochs, decreaseM = False, decreaseC = False, \
                        MStartEpoch = mStartEpoch, MEndEpoch = mEndEpoch, CStartEpoch = cStartEpoch, CEndEpoch = cEndEpoch)
            ga = Ga(r, popSize, registers, generationFunction = QASMPrimitiveInstruction, minSize = minSize, maxSize = maxSize)
            stats = ga.evolve(totalEpochs, probCrossover = CProb, probMutation= MProb, \
                        evaluationFunc = incDecExp.evaluationFunction, selectionFunc= incDecExp.rouletteWheelSelection, \
                        params = params)
        except Exception:
            traceback.print_exc()
        generateSummaries(stats = stats, experimentName="inc-MC")

    if(enabledExperiments[2]):
        
        try:
            # decreasing CProb, MProb with default roulette wheel selection but delta mutation func
            #params = GAParameters(MProb, CProb, totalEpochs, decreaseM = True, decreaseC = True, \
            #            MStartEpoch = mStartEpoch, MEndEpoch = mEndEpoch, CStartEpoch = cStartEpoch, CEndEpoch = cEndEpoch)
            
            #constant cprob, mprob with default roulette wheel but delta mutation func
            params = GAParameters(MProb, CProb, totalEpochs, decreaseM = False, decreaseC = False, \
                        MStartEpoch = mStartEpoch, MEndEpoch = mEndEpoch, CStartEpoch = cStartEpoch, CEndEpoch = cEndEpoch)
            ga = Ga(r, popSize, registers, generationFunction = QASMPrimitiveInstruction, minSize = minSize, maxSize = maxSize)
            stats = ga.evolve(totalEpochs, probCrossover = CProb, probMutation= MProb, \
                        evaluationFunc = incDecExp.evaluationFunction, selectionFunc= incDecExp.rouletteWheelSelection, \
                        mutationArgFunc = deltaMutationFunc, params = params)
        except Exception:
            traceback.print_exc()
        generateSummaries(stats = stats, experimentName="inc-rw")

    if(enabledExperiments[3]):
        
        try:
            # decreasing CProb, MProb with elitism selection but delta mutation func
            # params = GAParameters(MProb, CProb, totalEpochs, decreaseM = True, decreaseC = True, \
            # MStartEpoch = mStartEpoch, MEndEpoch = mEndEpoch, CStartEpoch = cStartEpoch, CEndEpoch = cEndEpoch, \
            # eliteRatio=eliteRatio)
            
            # constant CProb, MProb with elitism selection but delta mutation func
            params = GAParameters(MProb, CProb, totalEpochs, decreaseM = False, decreaseC = False, \
             MStartEpoch = mStartEpoch, MEndEpoch = mEndEpoch, CStartEpoch = cStartEpoch, CEndEpoch = cEndEpoch, \
             eliteRatio=eliteRatio)
            ga = Ga(r, popSize, registers, generationFunction = QASMPrimitiveInstruction, minSize = minSize, maxSize = maxSize)
            stats = ga.evolve(totalEpochs, probCrossover = CProb, probMutation= MProb, \
                        evaluationFunc = incDecExp.evaluationFunction, selectionFunc= incDecExp.elitismSelection, \
                        mutationArgFunc = deltaMutationFunc, params = params)
        except Exception:
            traceback.print_exc()
        generateSummaries(stats = stats, experimentName="inc-e")
    input("Press [enter] to continue.")


def decrementExperiment():
    """ 
    This is the main method for the decrement problem 
    This problem is phrased as follows:
    given a set of registers, we will create a prepare statement corresponding to a value in superposition, such as 
    0x4, the decrement op should output 50% for 0 and 5 correspondingly
    
    We will implement the experiment for 5 qubits (+1 qubit as scratch): 
    the first qubit is the min value within the circuit initially: x a[0] 
    (as in the prepareStatement)
    So the values in the lookup table are : 
    h a[1] (2^1) - {'00000': 0.5, '00010': 0.5}
    h a[2] (2^2) - {'00000': 0.5, '00100': 0.5}
    h a[3] (2^3) - {'00000': 0.5, '01000': 0.5}
    h a[4] (2^4) - {'00000': 0.5, '10000': 0.5}

    """
    #sets up entries in the lookup table
    lookupTable = [
        ('00000','00010', "x q[0];\nh q[1];\nrz(0.785398163397448) q[1];\n"),
        ('00000','00100', "x q[0];\nh q[2];\nrz(0.785398163397448) q[2];\n"),
        ('00000','01000', "x q[0];\nh q[3];\nrz(0.785398163397448) q[3];\n"),
        ('00000','10000', "x q[0];\nh q[4];\nrz(0.785398163397448) q[4];\n"),
    ]
    
    #this array makes it easy to control which experiment is to run by the sim
    enabledExperiments = [False,False,True,False]
    #default params
    params = GAParameters(MProb, CProb, totalEpochs)
    incDecExp = IncDecExperiment(registers, lookupTable, r, popSize)

    if(enabledExperiments[0]):
        # the default roulette wheel selection with a constant CProb, MProb, 80% retention rate            
        try:
            ga = Ga(r, popSize, registers, generationFunction = QASMPrimitiveInstruction, minSize = minSize, maxSize = maxSize)
            stats = ga.evolve(totalEpochs, probCrossover = CProb, probMutation= MProb, \
                        evaluationFunc = incDecExp.evaluationFunction, selectionFunc= incDecExp.rouletteWheelSelection, \
                        params = params)
        except Exception:
            traceback.print_exc()
        generateSummaries(stats = stats, experimentName="dec-none")
    
    if(enabledExperiments[1]):
        # decreasing CProb, MProb with default roulette wheel selection
        try:
            params = GAParameters(MProb, CProb, totalEpochs, decreaseM = True, decreaseC = True, \
                        MStartEpoch = mStartEpoch, MEndEpoch = mEndEpoch, CStartEpoch = cStartEpoch, CEndEpoch = cEndEpoch)
            ga = Ga(r, popSize, registers, generationFunction = QASMPrimitiveInstruction, minSize = minSize, maxSize = maxSize)
            stats = ga.evolve(totalEpochs, probCrossover = CProb, probMutation= MProb, \
                        evaluationFunc = incDecExp.evaluationFunction, selectionFunc= incDecExp.rouletteWheelSelection, \
                        params = params)
        except Exception:
            traceback.print_exc()
        generateSummaries(stats = stats, experimentName="dec-MC")

    if(enabledExperiments[2]):        
        try:
            # decreasing CProb, MProb with default roulette wheel selection but delta mutation func
            #params = GAParameters(MProb, CProb, totalEpochs, decreaseM = True, decreaseC = True, \
            #           MStartEpoch = mStartEpoch, MEndEpoch = mEndEpoch, CStartEpoch = cStartEpoch, CEndEpoch = cEndEpoch)
            
            #constant CProb, MProb with default roulette wheel selection but delta mutation func
            params = GAParameters(MProb, CProb, totalEpochs, decreaseM = False, decreaseC = False, \
                MStartEpoch = mStartEpoch, MEndEpoch = mEndEpoch, CStartEpoch = cStartEpoch, CEndEpoch = cEndEpoch)
            ga = Ga(r, popSize, registers, generationFunction = QASMPrimitiveInstruction, minSize = minSize, maxSize = maxSize)
            stats = ga.evolve(totalEpochs, probCrossover = CProb, probMutation= MProb, \
                        evaluationFunc = incDecExp.evaluationFunction, selectionFunc= incDecExp.rouletteWheelSelection, \
                        mutationArgFunc = deltaMutationFunc, params = params)
        except Exception:
            traceback.print_exc()
        generateSummaries(stats = stats, experimentName="dec-rw")

    if(enabledExperiments[3]):        
        try:
            # decreasing CProb, MProb with default roulette wheel selection but detal mutation func
            #params = GAParameters(MProb, CProb, totalEpochs, decreaseM = True, decreaseC = True, \
            # MStartEpoch = mStartEpoch, MEndEpoch = mEndEpoch, CStartEpoch = cStartEpoch, CEndEpoch = cEndEpoch, \
            # eliteRatio=eliteRatio)
            params = GAParameters(MProb, CProb, totalEpochs, decreaseM = False, decreaseC = False, \
                MStartEpoch = mStartEpoch, MEndEpoch = mEndEpoch, CStartEpoch = cStartEpoch, CEndEpoch = cEndEpoch, \
                eliteRatio=eliteRatio)
            ga = Ga(r, popSize, registers, generationFunction = QASMPrimitiveInstruction, minSize = minSize, maxSize = maxSize)
            stats = ga.evolve(totalEpochs, probCrossover = CProb, probMutation= MProb, \
                        evaluationFunc = incDecExp.evaluationFunction, selectionFunc= incDecExp.elitismSelection, \
                        mutationArgFunc = deltaMutationFunc, params = params)
        except Exception:
            traceback.print_exc()
        generateSummaries(stats = stats, experimentName="dec-e")
    input("Press [enter] to continue.")

def deltaMutationFunc(currentVal:float) -> float:
    # I think 2pi/32 is small enough for our experimentation purposes
    delta = 2 * pi / 32    
    """
    This methods generates a new thetha value based on the existing value + delta (instead of a random value)
    """
    if(r.random() > 0.5):
        return currentVal + delta
    return currentVal - delta

def elitismDefaultSelectionFunc(individuals: List[Chromosome], elitesRatio = 0.1) -> Tuple[List[Chromosome], List[float], bool, List[Chromosome]]:
    """
    This selection method performs elitism, here the nonElites ratio indicates that (by default), 90% of the individuals of
    the population are non-elites, or 10% of the individuals in the new population are elites.

    The algorithm starts by computing the fitness values (as usual) and store the values in tuples(indexToIndividual, fitness),
    it then selects the top X (determined by numOfElites) individuals as elites (preserved) while handing the rest to the selection
    method.
    """
    terminate = False
    retention = 0.8 - elitesRatio #TODO: we should probably encapsulate this in a class such as params
    threshold = 0.001
    numOfElites = ceil(elitesRatio * popSize)
    # 1. computes the probability (or fitness)
    maxvalue = 2 ** registers - 1
    aggregatedFitnesses = []
    fitnesses = []
    tuples = []
    total = 0.0
    fitness = -1
    currentIdx = 0
    for ind in individuals:
        probabilities = list(ind.getProbabilities().values())
        largest = 0
        # gets the prob of the last non 0 prob entry in the list. This is only applicable for the max binary string problem
        # since the largest value is the one with all 1111...
        for i in range(maxvalue,-1,-1):
            if(probabilities[i] > 0.0):
                # i is the corresponding value translate from binary since:
                # for 5 registers: 11111 = 31 and the largest offset is 31 in the array
                largest = i 
                break

        # the fitness value is the probability * value (state vector) for the state string that's closest to the max
        # value that can be represented by the qubits. For instance: given a 3 qubit problem, the max value is "111", so
        # if we obtained {000: 0.1, .... 100: 0.5, 101:0, 110:0, 111:0}, the fitness value is going to be int(100) * 0.5.
        fitness = probabilities[largest] * i

        # saves this in the chromosome
        ind.fitness = fitness

        # if the diff between the best fitness and 1.0 is close enough (threshold), we can terminate
        if(maxvalue - fitness < threshold):
            terminate = True

        # saves this in a tuplie within the fitnesses array
        tuples.append((currentIdx, fitness))
        fitnesses.append(fitness)
        #we are creating the aggregated fitness so that when we roll a random number, we just have to compare if it is <= a value in fitnesses[]
        total += fitness
        #creates the range of aggregated fitness for selection later
        aggregatedFitnesses.append(total) 
        #updates the current individual offset
        currentIdx += 1

    #sorts the tuplies
    ranked = sorted(tuples, key=lambda tup:tup[1], reverse=True)

    elites = []
    #keeps the elites
    for i in range(0, numOfElites):
        offset = ranked[i][0]
        elites.append(individuals[offset].getClone())       

    #we want to send of everything for selection
    newPop = Ga.defaultMakeSelection(r, popSize, individuals, aggregatedFitnesses, retention = retention)

    return newPop, fitnesses, terminate, elites

def weightedSelectionFunc(individuals: List[Chromosome], retention = 0.8) -> Tuple[List[Chromosome], List[float], bool]:
    """
    This selection method performs roulette wheel selection based on a weighted fitness value:
    prob1 * value of the first binary string + prob2 * value of the 2nd binary string + ...
    """
    terminate = False
    threshold = 0.001
    #computes the probability (fitness)
    aggregatedFitnesses = []
    fitnesses = []
    total = 0.0
    fitness = -1
    maxfitness = 2 ** registers - 1
    for ind in individuals:
        probabilities = list(ind.getProbabilities().values())
        # we need to iterate through all of the elements in the list and compute the fitness value, given that
        # this is a binary string problem, the values are arranged in 0, 1, 2,..., 2^numRegisters - 1 so it's straightforward
        # to compute this weighted probability
        for idx, prob in enumerate(probabilities):
            if (prob > 0.0):
                fitness += prob * idx
        ind.fitness = fitness
        # we can terminate when the weighted average fitness is >= 2^numRegisters - 1
        if(maxfitness -fitness < threshold):
            terminate = True
        fitnesses.append(fitness)
        total += fitness
        aggregatedFitnesses.append(total)
        fitness = 0.0
    return Ga.defaultMakeSelection(r, popSize, individuals, aggregatedFitnesses, retention = retention), fitnesses, terminate

def generateSummaries(stats : GAEpochStat, experimentName: str, params: GAParameters = None):
    print("Generating Summaries\n")
    #prints the graph
    LineGraph.showGraph(stats = stats, title="Fitness statistics", experiment=experimentName)
    #prints the best individuals in each epoch
    for i in range(0, len(stats)):
        print("Circuits of the best individual in epoch {0}, fitness {1}".format(i, stats[i].bestFitness))
        OpenQASMAdapter.printCircuit(OpenQASMConverter.convertPrimitivesToOpenQASM(registers, \
            stats[i].bestInd.instructions, prepareStatement = (params.prepareStatement if params != None else "")))
        print("Cases: {0}".format(stats[i].bestProb))

if __name__ == "__main__":
    #main() # the main method for the maxone problem
    #incrementExperiment() # the main method for the increment problem
    decrementExperiment() # the main method for the decrement problem