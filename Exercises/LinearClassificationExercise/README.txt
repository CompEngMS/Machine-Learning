The files in this folder contains Python implementations 
of some methods for linear classification: 
Least square, Fisher discriminants and SimplePerceptron
as well as interface with linear classifiers implemented
in the scikit learn library

Required Python libraries: numpy, matplotlib, sklearn

Run main.py with some options as described below.

usage: main.py [-h] [-n N] [--outliers] [-niter NITER] [-eta ETA] method

Linear Classification exercise

positional arguments:
  method        L=Least square, F=Fisher, p=SimplePerceptron, P=Perceptron,
                S=SVM

optional arguments:
  -h, --help    show this help message and exit
  -n N          size of data set
  --outliers    presence of outliers in the data set
  -niter NITER  SimplePerceptron: number of iterations
  -eta ETA      SimplePerceptron: eta learning factor

  
