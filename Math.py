import numpy as np
from collections import Sequence

import math


class Tangent_Type:
    def __init__(self, v, t=0.0):
        self.v = v
        self.t = t

    def Tangentify(input_func):
        def func(self, x):
            if not isinstance(x, Tangent_Type):
                x = Tangent_Type(x)
            return input_func(self, x)
        return func

    @Tangentify
    def __add__(self, x): 
        return Tangent_Type(self.v + x.v, self.t + x.t)

    __radd__ = __add__

    @Tangentify
    def __sub__(self, x):
        return Tangent_Type(self.v - x.v, self.t - x.t)
    
    @Tangentify
    def __rsub__(self, x):
        return Tangent_Type(x.v - self.v, x.t - self.t)

    @Tangentify
    def __mul__(self, x):
        return Tangent_Type(self.v*x.v, self.v*x.t + self.t*x.v)

    __rmul__ = __mul__

    @Tangentify
    def __truediv__(self, x):
        return Tangent_Type(self.v/x.v , self.t/x.v-(self.v*x.t)/(x.v*x.v))
    
    @Tangentify
    def __rtruediv__(self, x):
        return Tangent_Type(x.v/self.v , x.t/self.v-(x.v*self.t)/(self.v*self.v))

    @Tangentify
    def __pow__(self, x):
        return Tangent_Type(self.v**x.v, self.t*x.v*self.v**(x.v-1) + math.log(self.v)*self.v**x.v*x.t)


def Tangentify(input_func):
        def func(x):
            if not isinstance(x, Tangent_Type):
                x = Tangent_Type(x)
            return input_func(x)
        return func

@Tangentify
def cos(x):
    return Tangent_Type(math.cos(x.v), -x.t*(math.sin(x.v)))

@Tangentify
def sin(x):
    return Tangent_Type(math.sin(x.v), x.t*(math.cos(x.v)))

@Tangentify
def tan(x):
    return Tangent_Type(math.tan(x.v), x.t*(1+math.tan(x.v)**2)**2)

@Tangentify
def exp(x):
    return Tangent_Type(math.exp(x.v), x.t*math.exp(x.v))

@Tangentify
def log(x):
    return Tangent_Type(math.log(x.v), x.t/x.v)

def pow(x, y):
    if not isinstance(x, Tangent_Type):
            x = Tangent_Type(x)
    if not isinstance(y, Tangent_Type):
            y = Tangent_Type(y)
    return Tangent_Type(math.pow(x.v, y.v), x.t*y.v*x.v**(y.v-1) + math.log(x.v)*y.t*x.v**y.v)
