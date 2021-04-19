from abc import ABC, abstractmethod
from _collections_abc import generator
import enum
'''
This class describes the types that serves as input to the OpenQASM instructions
'''
from test.leakers import test_selftype
class BaseType(ABC):
    '''
    This returns an enum description of the type
    '''
    @abstractmethod
    def getType(self):
        pass
    
    '''
    This returns the value represented by this type
    '''
    @abstractmethod
    def getValue(self):
        pass
    
    '''
    This randomizes the value of this type of input based on a given rand
    '''
    @abstractmethod
    def randomize(self, rand):
        pass
    
    '''
    The toString method creates a string representation of this type
    '''
    @abstractmethod
    def toString(self):
        pass
