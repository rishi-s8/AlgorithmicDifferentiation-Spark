from Math import log, sin, cos
from AD import Tangent_Mode as TM
from AD_Distributed import Tangent_Mode as TMD
import numpy as np

def f(inputs):
    x = inputs[0]
    ncp = inputs[1].v
    ncs = inputs[2].v
    
    p_size = inputs[3].v
    p = inputs[4:4+p_size]
    assert p_size%ncs == 0, "Check input! Inconsistency in size of vector and ncs"

    dW = np.array(inputs[4+p_size:])
    assert (len(dW)/p_size)%ncp == 0, "Check input! Inconsistency in size of dW and ncp"
    dW = np.reshape(dW, (-1, p_size))
    
    s = 0
    for j in range(0, len(dW), ncp):
        x0 = x
        for J in range(j, j+ncp):
            for i in range(0, p_size, ncs):                
                dt = 1.0/p_size
                t = i*dt
                for I in range (i, i+ncs):          
                    x = x + dt*p[I]*sin(x*t) + p[I]*cos(x*t)*np.sqrt(dt)*dW[J][I]
                    t = t + dt
            s = s+x
            x = x0
    x = s/len(dW)
    return x

if __name__ == "__main__":
    # inputs for Monte Carlo: x, ncp, ncs, p_size, p_list_elements, dW_elements
    x = [3, 2, 2, 4, 1, 1, 1, 1, 0.23, 0.001, 0.1, 0.12, 0.011, 0.03, 0.3, 0.5]
    tangent_mode = TM(len(x),1,f)
    print(tangent_mode.computeFullJacobian(x))
    tangent_mode = TMD(len(x),1,f)
    print(tangent_mode.computeFullJacobian(x))