import findspark
findspark.init()
from pyspark import SparkConf, SparkContext

import numpy as np
from collections import Sequence
from Math import Tangent_Type

conf = SparkConf().setMaster("local").setAppName("Differentiation")
sc = SparkContext.getOrCreate(conf = conf)


class Tangent_Mode:
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
        