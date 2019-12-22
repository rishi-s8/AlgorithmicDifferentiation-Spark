import Math

def f(x):
    return x[0]*x[1]

if __name__ == "__main__":
    x = [3.0,5.0]
    tangent_mode = Math.Tangent_Mode_Distributed(2,1,f)
    print(tangent_mode.computeFullJacobian(x))