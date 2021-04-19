from typing import List, Tuple
from GAOP.GeneticAlgorithms.Chromosome import Chromosome
from GAOP.QiskitDrivers.OpenQASMAdapter import OpenQASMAdapter
from GAOP.QiskitDrivers.OpenQASMConverter import OpenQASMConverter
from GAOP.GeneticAlgorithms.Ga import Ga
from GAOP.Statistics.RandomWrapper import RandomWrapper
import math
class IncDecExperiment:
    def __init__(self, numOfRegisters:int, testVals: List[Tuple[str, str, str]], rand: RandomWrapper, popSize: int):
        """ 
        Sets up the paramters 
        The testVals work as follows:
        each tuple contains the first two values that should have 0.5 prob. 
        For instance, in an increment experiment, given the 3rd bit in superposition = 0x4, we should see a value of
        1 or 5 (in superposition) post increment since 0x4 in superposition is either 0 or 1, so if its 0, then the value 
        is 0 (thus, with a post increment value of 1), whereas if the value's of the 3rd bit is 1 (value of 4 = 2^2 from 
        2^0, 2^1, 2^2, etc etc), the value is 4 (thus with a post increment value of 5).
        The last entry is the prepareStatement to test out the probabilities. so what we want is to test out the circuit on
        all of the entries in the testVals with the 3rd element (prepareStatement) in the tuple
        """
        self.numOfRegisters = numOfRegisters
        self.testValues = testVals
        self.rand = rand
        self.popSize = popSize

    def evaluationFunction(self, numOfRegisters:int, collection:List[str], prepareStatement = None) -> Tuple[dict, dict]:
        """
        Inputs:
        This method has no use for the prepareStatement, numOfRegisters since they should come in from the Tuple from __init__,
        thus they are ignored.

        The evalution scheme:
        Since the total prob of each entries in the adds up to 1, the strongest individual is the one with all 1s, so the 
        max fitness value is: len(Tuple).

        Return values:
        data = unused, so {}
        proabilities = {fitness values for each cases in the testValues data structure}
        TODO: we might wanna consider those that are less fit later, we will see.
        """
        i = 0
        fitnesses = {}
        fitness = 0.0
        # evaluate the output for each tuple in the testValues list
        for tup in self.testValues:
            # tup(2) holds the prepare statement
            code = OpenQASMConverter.convertPrimitivesToOpenQASM(numOfRegisters, collection, tup[2])
            data, probabilities = OpenQASMAdapter.execute(code)
            # lookup the corresponding probabilities            
            fitnesses[i] = self.__getFitnessValue(probabilities[tup[0]]) + self.__getFitnessValue(probabilities[tup[1]])
            i+= 1
        return {}, fitnesses        
    
    def rouletteWheelSelection(self, individuals: List[Chromosome], retention = 0.8) -> Tuple[List[Chromosome], List[float], bool]:
        """
        The probabilities from individuals are now based on the cases, to get the total fitness, we will sum all of the tup[1]
        """
        terminate = False
        aggregatedFitnesses = []
        fitnesses = []
        total = 0.0
        fitness = -1
        #computes the aggregated fitness
        for ind in individuals:
            probs = ind.getProbabilities()
            fitness = sum(v for v in probs.values())
            ind.fitness = fitness
            if(not terminate): #terminate should only be set to true if it's false, this prevents overwritting an otherwise true.
                terminate = self.__checkTerminateCondition(ind.fitness)
            #saves in the fitness array
            fitnesses.append(fitness)
            # updates the aggregated fitness
            total += fitness
            aggregatedFitnesses.append(total)
        return Ga.defaultMakeSelection(self.rand, self.popSize, individuals, aggregatedFitnesses, retention = retention), fitnesses, terminate
    
    def elitismSelection(self, individuals: List[Chromosome], elitesRatio = 0.1) -> Tuple[List[Chromosome], List[float], bool]:
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
        numOfElites = math.ceil(elitesRatio * self.popSize)
        # 1. computes the probability (or fitness)
        maxvalue = len(self.testValues)
        aggregatedFitnesses = []
        fitnesses = []
        tuples = []
        total = 0.0
        fitness = -1
        currentIdx = 0
        for ind in individuals:
            #given that we may have couple of entries, we will get the sum of fitness in this case
            fitness = sum(v for v in ind.getProbabilities().values())
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
        newPop = Ga.defaultMakeSelection(self.rand, self.popSize, individuals, aggregatedFitnesses, retention = retention)
        return newPop, fitnesses, terminate, elites

    def __checkTerminateCondition(self, val:float) -> bool:
        """
        The max value is #_of_entries * (0.5 + 0.5) = #_of_entries
        """
        if(val == len(self.testValues)):
            return True
        return False

    def __getFitnessValue(self, val:float) -> float:
        if(val >= 0.5):
            return (val * -1.0 + 1)        
        return val

        