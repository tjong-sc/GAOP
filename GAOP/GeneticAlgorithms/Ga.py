from GAOP.Representations.QASMPrimitiveInstruction import QASMPrimitiveInstruction
from GAOP.QiskitDrivers.OpenQASMConverter import OpenQASMConverter
from GAOP.QiskitDrivers.OpenQASMAdapter import OpenQASMAdapter
from GAOP.GeneticAlgorithms.Population import Population
from GAOP.GeneticAlgorithms.Chromosome import Chromosome
from GAOP.Statistics.GAEpochStat import GAEpochStat
from GAOP.Statistics.RandomWrapper import RandomWrapper
from GAOP.GeneticAlgorithms.GAParameters import GAParameters
import traceback
import math
import sys
import operator
from typing import List, Tuple

class Ga:
    """
        This class simplifies the creation and manamagement of a GA
        The default settings of this class searches for a set of OpenQASM instructions that maximizes the
        binary string problem from a given set of registers. (i.e. max probability in "111111..." string)
    """
    def __init__(self, rand: RandomWrapper, size:int, numOfRegisters : int, generationFunction : callable, minSize : int, maxSize : int, \
                 arcFunc: callable = None):
        """ The constructor creates an initial population based on the given generation function """
        self.arcFunc = arcFunc
        self.size = size
        self.rand = rand
        self.generationFunction = generationFunction
        self.numOfRegisters = numOfRegisters
        try:
            #creates the initial population with the parameters supplied
            self.pop = Population(rand, numOfRegisters, generationFunction = generationFunction, \
                chromosomeMinSize = minSize, chromosomeMaxSize = maxSize, size = size)
        except:
            traceback.print_exc()

    def evolve(self, totalEpochs : int, probCrossover: float, probMutation: float, selectionFunc: callable = None, \
        terminationFunc: callable = None, crossOverFunc : callable = None, mutationFunc: callable = None, \
        mutationCmdFunc: callable = None, mutationArgFunc: callable = None, evaluationFunc: callable = None, \
            params:GAParameters = None ) \
            -> List[GAEpochStat]:
        """ 
        The evolve method loops thru the GA population until a solution is found (100% probability) or the num epoch is reached 
        Parameters for the various callables are as follows:
            selectionFunc (if any, else revert to what's defined in this file <TODO>) - 
                selectionFunc(population) -> list of Chromosomes
            crossoverFunc (if any, else revert to what's defined in Population.py - 
                MultiPointCrossover()): crossOverFunc(chromosome, chromosome)
            mutationFunc (if any, else revert to what's defined in Population.py - 
                UniformMutation()): mutationFunc(chromosome)
            terminationFunc (if any, else revert to "a solution is found" or num of epoch is reached.)
            probUpdateFunc (if any, else do nothing) - this allows us to provide a mathematical function to reduce the 
                mutation and crossover probs overtime
        """
        elites = None
        #use default if nothing is provided
        if(selectionFunc == None):
            selectionFunc = self.__defaultRouletteWheelSelectionMaxProb

        if(crossOverFunc == None):
            crossOverFunc = self.__defaultMultipointCrossover

        if(mutationFunc == None):
            mutationFunc = self.__defaultMutation

        stats = [] # an array of statistics class
        currentEpoch = 0
        while True:
            print("********************* Initial Population in epoch {0}******************************".format(currentEpoch))
            #self.pop.printIndividuals("\t")

            # the order is as follows:
            # 1. evaluate population fitness            
            # 1a) converts the population into code 
            selectedParents = []
            individuals = self.pop.getIndividuals()
            for ind in individuals:
                openQASMPreparestatement = (params.openQASMPrepareStatement if params != None else "")
                if(evaluationFunc == None):
                    code = OpenQASMConverter.convertPrimitivesToOpenQASM(numOfRegisters = self.numOfRegisters, \
                    collection=ind.instructions, prepareStatement = openQASMPreparestatement)
                    # 1b) runs the code via the adapter
                    data, probabilities = OpenQASMAdapter.execute(code)
                    # records the fitnesses in ind (chromosome type)
                    ind.setProbabilities(probabilities)
                else:
                    data, probabilities = evaluationFunc(numOfRegisters = self.numOfRegisters, \
                        collection = ind.instructions, prepareStatement = openQASMPreparestatement)
                    ind.setProbabilities(probabilities)

            if(params != None and params.eliteRatio > 0.0):
                # 2. selection, note that the population may be incomplete, so always fill up the population with random individuals
                selectedParents, fitnesses, terminate, elites = selectionFunc(individuals, elitesRatio = params.eliteRatio)
            else:
                selectedParents, fitnesses, terminate = selectionFunc(individuals)

            # finds the largest value and index in fitnesses - best individual
            stats.append(self.__gatherStats(individuals, fitnesses, currentEpoch))

            if (terminate):
                break

            # 3. crossover
            selectedParents = crossOverFunc(selectedParents, probCrossover)

            # 4. mutation func
            selectedParents = mutationFunc(individuals = selectedParents, probMutation = probMutation,\
                commandfunc = mutationCmdFunc, argumentFunc = mutationArgFunc)

            # 4b. fills up the population with elites if any
            if (elites != None and len(elites) > 0):
                for i in range(0, len(elites)):
                    selectedParents.append(elites[i])

            # 5. fills up the population with random individuals
            self.__fillsUpPopulation(selectedParents)

            # 6. updates the population
            self.pop.setIndividuals(selectedParents)
            
            print("+++++++++++++++ Eventual Pop of epoch {0}+++++++++++++++".format(currentEpoch))
            self.pop.printIndividuals("\t")
            print("+++++++++++++++ End of epoch{0}+++++++++++++++".format(currentEpoch))
            # 7. Checks for termination condition: epoch limit is reached
            if currentEpoch >= totalEpochs:
                #all done
                break

            # 8. performs update on the paramters if necessary
            if(params != None):
                probCrossover = params.getCRate(currentEpoch)
                probMutation = params.getMRate(currentEpoch)

            currentEpoch += 1
        return stats

    def __gatherStats(self, inds: List[Chromosome], fitnesses: List[float], epoch: int) -> 'GAEpochStat':
        bestFit = max(fitnesses)
        bestIdx = fitnesses.index(bestFit)
        bestProb = inds[bestIdx].probabilities
        minFit = min(fitnesses)
        minIdx = fitnesses.index(minFit)
        minProb = inds[minIdx].probabilities
        avgFit = sum(fitnesses)/len(fitnesses)
        return GAEpochStat(inds[bestIdx].getClone(), inds[minIdx].getClone(), bestProb, minProb,\
            bestFit, minFit, avgFit, epoch)

    def printAllChromosomes(self):
        for ind in self.pop.individuals:
            print(ind.toString("t"))

    def __printChromosomes(self, individuals: List[Chromosome]) -> None:
        for ind in individuals:
            print(ind.toString("\t"))

    def __fillsUpPopulation(self, individuals: List[Chromosome]) -> List[Chromosome]:
        """
        This method can be used to fill up the remaining population with random individuals
        """
        # 3. remaining 20%
        while len(individuals) < self.size:
            individuals.append(Chromosome(self.rand, self.numOfRegisters, self.generationFunction, self.size, self.arcFunc))
            if len(individuals) >= self.size:
                break

    def __defaultMultipointCrossover(self, individuals: List[Chromosome], probCrossover) -> List[Chromosome]:
        """
        This crossover method performs multipoint crossover at 65% probability (65-81% are good based on lit review)
        It will: pick two parents, roll dice to determine if we should crossover then writes them to the new population
        """
        newPop = []
        size = len(individuals) - 1
        while True:
            # pick two parents
            parents = self.rand.sample(individuals, 2)
            # roll the dice, then perform crossover
            if(self.rand.random() < probCrossover):
                # crossover is performed
                new1, new2 = parents[0].instructionsMultipointCrossover(parents[1])
            else:
                new1 = parents[0].getClone()
                new2 = parents[1].getClone()
            newPop.append(new1)
            newPop.append(new2)
            if(len(newPop) > size):
                break
        return newPop

    def __defaultMutation(self, individuals: List[Chromosome], probMutation, commandfunc:callable = None, \
        argumentFunc:callable = None) -> List[Chromosome]:
        """
        This default mutation method performs mutation at the rate of 0.1% (or 0.001)
        What it will do is it will go through each instructions than roll the dice and perform mutation on either instruction/register/argument
        in a random manner
        """
        size = len(individuals)
        for i in range(0, size):
            # The uniform Mutation function automatically rolls the dice to determine if the current 
            # individual should undergo mutation
            individuals[i].uniformMutation(mutationProbability = probMutation, \
                commandfunc = commandfunc, argumentFunc = argumentFunc)

        return individuals

    @classmethod
    def safe_divide(cls, numerator:float, denominator:float) -> float:
        """
        TODO: This should sit in a utility package or something
        """
        fitness = 1.0
        try:
            # gets the distance of the entries with the largest probability then computes the 1/distance to get the fitness
            fitness = numerator/denominator
        except ZeroDivisionError:
            fitness = 1.0
        return fitness

    @staticmethod
    def defaultMakeSelection(rand: RandomWrapper, size: int, individuals: List[Chromosome], aggregatedFitnesses: List[float], retention = 0.8) -> List[Chromosome]:
        newIndividuals = []
        #selection 80%
        toRemain = math.floor(len(individuals) * retention)
        total = aggregatedFitnesses[-1]
        for i in range(0, toRemain):
            selected = False
            threshold = rand.uniform(0, total)
            for j in range(0, size):
                #picks an individual when it's aggregated fitness exceeds the threshold
                if(aggregatedFitnesses[j] > threshold):
                    newIndividuals.append(individuals[j].getClone())
                    selected = True
                    break
            if(not selected):
                #if not selected, we will grab the last individual from the list
                newIndividuals.append(individuals[-1].getClone())
        return newIndividuals

    def __defaultRouletteWheelSelectionMaxProb(self, individuals: List[Chromosome], retention = 0.8) -> Tuple[List[Chromosome], List[float], bool]:
        """
        This default selection method combines the probability of all of the individuals.
        (DEFAULT behavior) - The fitness it tries to evaluate is to find a set of instructions that turns all of the qubits into 1s.
        It doesn't do much, and it tries to retain 80% old individuals
        outputs:
            fitness - corresponds to the original individuals (instead of the selected)
            terminate - represents the termination condition
        """
        terminate = False
        threshold = 0.001
        # 1. computes the probability (or fitness)
        maxvalue = 2 ** self.numOfRegisters - 1
        aggregatedFitnesses = []
        fitnesses = []
        total = 0.0
        fitness = -1
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

            # saves this in the fitnesses array
            fitnesses.append(fitness)
            #we are creating the aggregated fitness so that when we roll a random number, we just have to compare if it is <= a value in fitnesses[]
            total += fitness
            #creates the range of aggregated fitness for selection later
            aggregatedFitnesses.append(total) 

        return Ga.defaultMakeSelection(self.rand, self.size, individuals, aggregatedFitnesses, retention = retention), fitnesses, terminate

