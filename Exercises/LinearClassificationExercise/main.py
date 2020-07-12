# Test linear classification methods
# Luca Iocchi, 2014-2018

import argparse
from generateData import generateData
from plotData import *
from leastSquare import LeastSquare
from FisherDiscriminant import FisherDiscriminant
from perceptron import SimplePerceptron
from sklearn import svm
from sklearn.linear_model import Perceptron

classifier = None # classifier object

ClassifierMap = {
    'L': [LeastSquare, 'Least Square'], 
    'F': [FisherDiscriminant, 'Fisher Discriminant'], 
    'p': [SimplePerceptron, 'Simple Perceptron'], 
    'P': [Perceptron, 'Perceptron'], 
    'S': [svm.LinearSVC, 'SVM']
    }

# Main

np.random.seed(2018)

parser = argparse.ArgumentParser(description='Linear Classification exercise')
parser.add_argument('method', type=str, help='L=Least square, F=Fisher, p=SimplePerceptron, P=Perceptron, S=SVM')
parser.add_argument('-n', type=int, help='size of data set', default=100)
parser.add_argument('--outliers', help='presence of outliers in the data set', action='store_true')
parser.add_argument('-niter', type=int, help='SimplePerceptron: number of iterations', default=100)
parser.add_argument('-eta', type=float, help='SimplePerceptron: eta learning factor', default=0.01)


args = parser.parse_args()

if args.method not in ClassifierMap.keys():
    print "Method ",args.method," not available."
    sys.exit(1)

# generate data set (param: outliers=true/false)
[X,t] = generateData(args.n,args.outliers)

# run the chosen method
classifier = ClassifierMap[args.method][0]()

# set parameters
if (args.method=='p'):  # SimplePerceptron
    classifier.eta = args.eta
    classifier.niter = args.niter

# train the classifier
classifier.fit(X,t)

# show results
plotResult(X,t,classifier,ClassifierMap[args.method][1])

