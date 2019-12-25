import numpy as np
from collections import Sequence
from Math import Tangent_Type


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
        directionalDerivative = np.zeros(shape=(self.n_output,), dtype="float64")
        y = self.f(x)
        if isinstance(y, Sequence):
            for j in range(self.n_output):
                directionalDerivative[j] = y[j].t
        else:
            directionalDerivative[0] = y.t
        return directionalDerivative, y.v
