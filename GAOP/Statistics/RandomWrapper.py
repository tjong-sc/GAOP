from datetime import datetime
from typing import List
from numpy.random import RandomState

class RandomWrapper:
    """ This is a wrapper for the numpy random object"""
    rand = None

    def __init__(self, seed = None):
        if(seed is None):
            seed = int(datetime.utcnow().timestamp())
            print("************* Seed is: {0}\n".format(seed) )
            self.rand = RandomState(seed)
        else:
            self.rand = RandomState(seed)

    def randint(self, low: int, high: int) -> int:
        """ low and high are both inclusive and exclusive respectively"""
        return self.rand.randint(low, high)

    def uniform(self, low: float, high:float ) -> float:
        return self.rand.uniform(low, high)

    def random(self) -> float:
        return self.rand.uniform(0.0, 1.0)

    def sample(self, elements: List, size: int) -> List:
        return self.rand.choice(elements, size = size, replace = False)