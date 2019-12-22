import findspark
findspark.init()
from pyspark import SparkConf, SparkContext

import numpy as np
from collections import Sequence

conf = SparkConf().setMaster("local").setAppName("Differentiation")
sc = SparkContext.getOrCreate(conf = conf)
import math

def cos(x: Tangent_Type):
    return Tangent_Type(math.cos(x.v), -x.t*(math.sin(x.v)))

def sin(x: Tangent_Type):
    return Tangent_Type(math.sin(x.v), x.t*(math.cos(x.v)))

def tan(x: Tangent_Type):
    return Tangent_Type(math.tan(x.v), x.t*(1+math.tan(x.v)**2)**2)

def exp(x: Tangent_Type):
    return Tangent_Type(math.exp(x.v), x.t*math.exp(x.v))

def log(x: Tangent_Type):
    return Tangent_Type(math.log(x.v), x.t/x.v)


class Tangent_Type:
    def __init__(self, v, t=0.0):
        self.v = v
        self.t = t
    
    def __add__(self, x): 
        return Tangent_Type(self.v + x.v, self.t + x.t)

    __radd__ = __add__

    def __sub__(self, x):
        return Tangent_Type(self.v - x.v, self.t - x.t)
    
    __rsub__ = __sub__

    def __mul__(self, x):
        return Tangent_Type(self.v*x.v, self.v*x.t + self.t*x.v)

    __rmul__ = __mul__

    def __truediv__(self, x):
        return Tangent_Type(self.v/x.v , self.t/x.v-(self.v*x.t)/(x.v*x.v))

    __rtruediv__ = __truediv__


class Tangent_Mode:
    def __init__(self, n_input, n_output, f):
        self.n_input = n_input
        self.n_output = n_output
        self.f = f
    
    def computeFullJacobian(self, x):
        x = list(map(Tangent_Type, x))
        Jacobian = np.zeros(shape=(self.n_output, self.n_input), dtype="float64")
        for i,xi in enumerate(x):
            xi.t = 1.0
            y = self.f(x)
            if isinstance(y, Sequence):
                for j in range(self.n_output):
                    Jacobian[j,i] = y[j].t
            else:
                Jacobian[0,i] = y.t
            xi.t = 0.0
        return Jacobian, y.v
    
    def computeDerivative(self, x, x_t):
        x = list(map(Tangent_Type, x))
        for i, xi_t in enumerate(x_t):
            x[i].t = xi_t
        directionalDerivative = np.zeros(shape=(self.n_output, self.n_input), dtype="float64")
        y = self.f(x)
        if isinstance(y, Sequence):
            for j in range(self.n_output):
                directionalDerivative[j,i] = y[j].t
        else:
            directionalDerivative[0,i] = y.t
        return directionalDerivative, y


class Tangent_Mode_Distributed:
    def __init__(self, n_input, n_output, f):
        self.n_input = n_input
        self.n_output = n_output
        self.f = f
    
    def computeFullJacobian(self, x):
        self.x = list(map(Tangent_Type, x))
        Jacobian = np.zeros(shape=(self.n_output, self.n_input), dtype="float64")
        x_t = [0]*self.n_input
        tangent_rdd = sc.parallelize([])
        for i in range(len(x)):
            x_t[i] = 1
            tangent_rdd=tangent_rdd.union(sc.parallelize([x_t]))
            x_t[i] = 0
        
        jacobian_rdd = tangent_rdd.map(self.computeDerivative)
        Jacobian = jacobian_rdd.collect()
        return Jacobian
    
    def computeDerivative(self, x_t):
        loc_one = -1
        for i, xi_t in enumerate(x_t):
            if xi_t == 1:
                loc_one = i + 1
            self.x[i].t = xi_t
        directionalDerivative = np.zeros(shape=(self.n_output,), dtype="float64")
        y = self.f(self.x)
        if isinstance(y, Sequence):
            for j in range(self.n_output):
                directionalDerivative[j] = y[j].t
        else:
            directionalDerivative[0] = y.t
        return loc_one, directionalDerivative
        