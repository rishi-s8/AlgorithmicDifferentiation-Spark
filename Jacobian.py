from Math import log
from AD import Tangent_Mode as TM
from AD_Distributed import Tangent_Mode as TMD

def f(x):
    return log(x[0]*x[1])

if __name__ == "__main__":
    x = [3.0,5.0]
    tangent_mode = TM(2,1,f)
    print(tangent_mode.computeFullJacobian(x))
    tangent_mode = TMD(2,1,f)
    print(tangent_mode.computeFullJacobian(x))