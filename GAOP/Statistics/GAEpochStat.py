from GAOP.GeneticAlgorithms.Chromosome import Chromosome
class GAEpochStat:
    """ This is a simple data sturcture to capture various population stats """
    def __init__(self, bestInd: Chromosome, worstInd:Chromosome, bestProb:dict, worstProb:dict, \
        bestFitness: float, worstFitness: float, avgFitness:float, epoch: int):
        self.bestInd = bestInd
        self.worstInd = worstInd
        self.bestFitness = bestFitness
        self.worstFitness = worstFitness
        self.avgFitness = avgFitness
        self.epoch = epoch
        self.bestProb = bestProb
        self.worstProb = worstProb
