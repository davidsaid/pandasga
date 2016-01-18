from core import GABaseObject

import numpy as np

class BinarySegment(GABaseObject):
    def __init__(self,
                 name,
                 number_of_bits,
                 formatter=None):
        self.number_of_bits = number_of_bits
        if formatter is None:
            formatter = lambda x : self.to_binary(x)
        super(BinarySegment, self).__init__(name=name,
                                        formatter=formatter)
        
    def to_binary(self, x):
        return '0b' + np.binary_repr(x, self.number_of_bits)
    
    def crossover(self, a, b):
        cross_point = np.random.randint(self.number_of_bits + 1)
        return self.head(a, cross_point) + self.tail(b, cross_point)
    
    def mutate(self, c):
        return long(c) ^ (1<<np.random.randint(self.number_of_bits))
    
    def tail(self, c, cross_point):
        return long(c) & cross_point
    
    def head(self, c, cross_point):
        return long(c) - self.tail(c, cross_point)
    
    def random_value(self):
        return np.random.randint(2**self.number_of_bits)
    
    def max_value(self):
        return 2**self.number_of_bits -1