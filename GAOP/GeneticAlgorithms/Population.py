from GAOP.Representations.QASMPrimitiveInstruction import QASMPrimitiveInstruction
from GAOP.QiskitDrivers.OpenQASMConverter import OpenQASMConverter
from GAOP.GeneticAlgorithms.Chromosome import Chromosome
from GAOP.Representations.BasicInstruction import Symbols
from GAOP.Statistics.RandomWrapper import RandomWrapper
from typing import List
import sys

class Population:
    """
    This class represents the GA population, the default parameters are as follows:
    inputs:
    size = 10 -> individuals
    probMutation = 0.1 -> 10%
    probCrossover = 0.1 -> 10%
    epoch = 30 cycles
    """
    def __init__(self, rand: RandomWrapper, sizeOfRegisters:int, generationFunction:callable, chromosomeMinSize:int, chromosomeMaxSize:int, \
                 size:int = 10, probMutation:float = 0.1, probCrossover:float = 0.1, epoch:int = 30, \
                 arcFunc:callable = None):
        self.rand = rand
        self.arcFunc = arcFunc
        self.individuals = []
        self.size = size
        self.probMutation = probMutation
        self.probCrossover = probCrossover
        self.epoch = 30
        self.numOfRegisters = sizeOfRegisters
        self.__generatePopulation(generationFunction, sizeOfRegisters, chromosomeMinSize, chromosomeMaxSize, arcFunc = self.arcFunc)

    def printIndividuals(self, prefix:str = "") -> None:
        for ind in self.individuals:
            print(ind.toString(prefix))

    def printIndividual(self, indOffset:int, prefix:str = "") -> None:
        """ prints the individual based on a given prefix, empty string by default"""
        print(self.individuals[indOffset].toString(prefix))

    def getIndividuals(self) -> List[Chromosome]:
        """ retrieves the collection of all of the chromosomes """
        return self.individuals

    def getNumberOfRegisters(self) -> int:
        return self.numOfRegisters

    def setIndividuals(self, chromosomes: List[Chromosome]) -> None:
        self.individuals = chromosomes

    def __generatePopulation(self, generationFunction:callable, sizeOfRegisters:int, minSize:int, maxSize:int, arcFunc: callable = None) -> None:
        """ This method generates a collection of individuals based on the given set of params """

        #generates the individuals within a population
        for i in range(0, self.size):
            currentSize = self.rand.randint(minSize, maxSize)
            ind = Chromosome(self.rand, sizeOfRegisters, generationFunction, currentSize, arcFunc = arcFunc)
            self.individuals.append(ind)