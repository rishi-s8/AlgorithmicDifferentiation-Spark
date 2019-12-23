# Algorithmic Differentiation using Apache Spark

An efficient way to compute the derivative of large computer programs in a distributed manner. 
#### Why we need Map-Reduce?
The computation of the Jacobian Matrix can be done without Map-Reduce. But computing it iteratively for each Cartesian basis vector, without map-reduce is extremely slow. Moreover, this is only feasible for the computations having less number of dimensions. In real life numerical calculations, the number of dimensions is quite large and it's not practically possible to compute the Jacobian iteratively for each subroutine.

## Installation: Setting up Spark 
### Python 3.8 currently does not support PySpark 
Downgrade your python version or create a virtual environment with Python 3.7.

Then use ```pip install pyspark``` and you are good to go.
