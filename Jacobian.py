from Math import *
from AD import Tangent_Mode
from AD_Distributed import Tangent_Mode_Distributed

def f(x):
    return log(x[0]*x[1])

if __name__ == "__main__":
    x = [3.0,5.0]
    tangent_mode = Tangent_Mode_Distributed(2,1,f)
    print(tangent_mode.computeFullJacobian(x))