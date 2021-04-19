import matplotlib.pyplot as plt 
import matplotlib
from GAOP.Statistics.GAEpochStat import GAEpochStat
from typing import List, Tuple
import platform
class LineGraph:
    """ this is a simple class that plots a multiline graph and labels"""
    def __init__(self):
        # do nothing
        pass

    @classmethod
    def showGraph(cls, stats: List[GAEpochStat], title: str, experiment: str) -> None :
        # this is a temporary fix to get around problem with
        # matplotlib in linux
        #if(platform.system() == "Linux"):
        #    matplotlib.use("TkAgg")
        """ 
        This class method plots the ga epoch stat for best ind, worst ind, best fitness, worst fitness, avg fitness and epoch numbers 
        x axis = epoch
        y axis = fitness values
        """
        bestfit, worstfit, avgfit, epochs = cls.__getSeries(stats)
        plt.plot(epochs, bestfit, label="{0} Best fitness".format(experiment))
        plt.plot(epochs, worstfit, label="{0} Worst fitness".format(experiment))
        plt.plot(epochs, avgfit, label="{0} Avg fitness".format(experiment))
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Fitness")
        plt.legend()
        plt.draw_all()
        plt.ion()
        plt.show()

    @classmethod
    def __getSeries(cls, stats: List[GAEpochStat]) -> Tuple[List[float], List[float], List[float], List[int]] :
        bestfit = [i.bestFitness for i in stats]
        worstfit = [i.worstFitness for i in stats]
        avgfit = [i.avgFitness for i in stats]
        epochs = [i.epoch for i in stats]
        return bestfit, worstfit, avgfit, epochs