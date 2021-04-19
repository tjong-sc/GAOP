from abc import ABC, abstractmethod
from _collections_abc import generator
#from test.leakers import test_selftype
import enum
import math
from GAOP.Statistics.RandomWrapper import RandomWrapper

class BasicInstruction(ABC):

    @property
    def __debug(self):
        return True
    
    @property
    @abstractmethod
    def defaultArgumentSeperator(self):
        pass
    
    @property
    @abstractmethod
    def defaultCommandSeperator(self):
        pass
    
    @property
    @abstractmethod
    def stringPatternWithNoArgument(self):
        pass
    
    @property
    @abstractmethod
    def stringPatternWithArgument(self):
        pass
        
    '''
    The getCommand method returns the command token of this instruction
    '''
    @abstractmethod
    def getCommand(self):
        pass
    
    '''
    The getParameters method returns the input parameters (0 or more in the form of an array) 
    for this instruction
    '''
    @abstractmethod
    def getInputs(self):
       pass
   
    '''
    The randomize method randomizes the inputs based on a given seeded random number generator
    Input parameters:
        rand : random number generator
        numQR: total number of quantum registers
    '''
    @abstractmethod
    def randomizeArgument(self, rand, numQR):
       pass
   
    '''
    The randomize method randomizes the register referred to by the instruction
    '''
    @abstractmethod
    def randomizeRegister(self, rand, numQR):
       pass
   
    '''
    The randomizeCommand method allows one to *mutate* the command based on a certain indicator function
    which may deem certain sets more favorable than the other, perhaps. This method automatically updates
    the necessary arguments when a command is changed.  
    Inputs:
        cmdfunc: parameterless command mutation indicator function
        argfunc: 1 parameter - argument mutation indicator function
        doTruncation: For instance, if the original function expects 3 arguments,
        but the new one expects only 1, it will truncate the extra two arguments. On the other hand, if the option
        is false, it will try to find a matching command based on the number of inputs and arguments (being more 
        gentle).
    '''
    @abstractmethod
    def randomizeCommand(self, cmdfunc, argfunc, doTruncation):
        pass
    '''
    The toString method creates a string out of this instruction. This method can be used to print human
    readable strings or send to the QiskitAdapter for execution
    '''
    @abstractmethod
    def toString(self):
        pass
    '''
    The default argument function simply ignores the val parameter.
    '''
    @abstractmethod
    def defaultArgumentFunc(self, val):
        pass
'''
This enumeration class is used to represent the types of the data represented by chromosomes.
Format: Given crz (1.1223) q[1]
            crz = instruction
            q[1] = register  
            1.1223 = argument
'''
class Symbols(enum.Enum):
    INSTRUCTION = 0
    REGISTER = 1
    ARGUMENT = 2    