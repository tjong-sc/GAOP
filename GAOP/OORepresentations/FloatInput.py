from GAOP.Statistics.RandomWrapper import RandomWrapper
import sys
from GAOP.Instructions.BaseType import BaseType
from math import radians
from _asyncio import Future

class FloatInput(BaseType):
    '''
    TODO: for now we are allowing -inf to +inf of float values, we might wanna control their range into radians
    in the Future
    '''
    def __init__(self, rand):
        pass