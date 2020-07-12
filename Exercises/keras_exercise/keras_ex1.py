import sys, os
import numpy as np
import argparse

from keras import backend

from keras_nn import *

datasets_dir = 'datasets'

def loaddataset(dsname, outputsize=1):
    fds = os.path.join(datasets_dir,'%s.csv' %(dsname))
    dataset = np.loadtxt(fds, delimiter=",")
    inputsize = len(dataset[0])-outputsize
    X = dataset[:,0:inputsize]
    Y = dataset[:,inputsize:]
    return [X,Y]


### main ###
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Keras/TF ANN examples')
    parser.add_argument('problem', type=str, help='problem (e.g., xor, nl6, noisyxor, binaryid)')
    parser.add_argument('-hidden', type=int, help='hidden units', default=-1)
    parser.add_argument('-model', type=str, help='model filename', default='')
    parser.add_argument('-seed', type=int, help='random seed', default=0)
    parser.add_argument('-niter', type=int, help='iterations', default=-1)
    parser.add_argument('-lrate', type=float, help='learning rate', default=-1)
    args = parser.parse_args()

    nhidden = 2
    if (args.problem=='xor' or args.problem=='nl6' or args.problem=='noisyxor'):
        [X,Y] = loaddataset(args.problem)
    elif (args.problem=='binaryid'):
        [X,Y] = loaddataset(args.problem,8)
        nhidden = 3
    else:
        print("Problem %s undefined." %args.problem)
        sys.exit(0)

    if args.hidden>0:
        nhidden = args.hidden
    ninput=len(X[0])
    noutput=len(Y[0])
    fnmodel = ''
    model = None
    if args.model!='':
        model = loadmodel(args.model)
        v = args.model.split('_')
        rs = int(v[1])
    else:
        if args.seed==0:
            x = datetime.datetime.now()
            rs = int(x.strftime("%y%m%d%H%M"))
        else:
            rs = args.seed

    np.random.seed(rs)
    if model==None:
        if (args.lrate>0):
            model = createmodel(ninput,noutput,nhidden,args.lrate)
        else:
            model = createmodel(ninput,noutput,nhidden)

    if (args.lrate>0):
        # set learning rate
        backend.set_value(model.optimizer.lr, args.lrate)
        print("New learning rate: %f" %backend.get_value(model.optimizer.lr))


    print("random seed: %d" %rs)
    [acc, loss, run] = train(model,X,Y,niter=args.niter)
    fnmodel = "%s_%d_%d_%03d" %(args.problem,rs,int(acc*100),int(loss*1000))
    print_solution(model,X)    
    savemodel(model,fnmodel)
