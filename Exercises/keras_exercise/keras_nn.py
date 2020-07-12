import sys, os
import numpy as np
import datetime
import argparse
import atexit

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

models_dir = 'models'

def createmodel(ninput,noutput,nhidden,lrate=1e-3):
    model = Sequential()
    if noutput==1:
        model.add(Dense(nhidden, input_dim=ninput, activation='relu'))
        model.add(Dense(noutput, activation='linear'))
    else:
        model.add(Dense(nhidden, input_dim=ninput, activation='sigmoid'))
        model.add(Dense(noutput, activation='sigmoid'))
    sgd = SGD(lr=lrate)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    print("Created new model i:%d h:%d o:%d" %(ninput,noutput,nhidden))
    return model

def train(model,X,Y,niter=-1):
    np.set_printoptions(precision=3, formatter={'float':lambda x: '%6.3f' %x}, suppress=True)

    d = 100
    c = 0
    run = True
    current_loss = 1.0
    current_acc = 0.0
    batch_size=4
    
    lastdd = np.zeros(len(Y[0]))
    while ((niter<0 or c<niter) and run):
        try:
            h = model.fit(X, Y, batch_size=batch_size, epochs=d*batch_size, verbose=0)
        except KeyboardInterrupt:
            run = False
        c += d
        current_loss = h.history['loss'][len(h.history['loss'])-1]
        current_acc = h.history['acc'][len(h.history['acc'])-1]
        print("Iteration %d - Accuracy %.2f Loss %f" 
            %(c,current_acc,current_loss))
        yp = model.predict(X)
        noutput = len(Y[0])
        if noutput==1:
            print("Test\n%s" %(np.transpose(yp)))
        else:
            for i in range(0,noutput):
                m = max(yp[i])
                dd = m-yp[i][i]
                if (dd>0):
                    td = dd - lastdd[i]
                    if (td>0):
                        ch = '+'
                    elif (td<0):
                        ch = '-'
                    else:
                        ch = '='
                    print("   %.3f -> %.3f  |  %.3f %c" %(yp[i][i],m,dd,ch))
                    lastdd[i] = dd
                else:
                    print("   %.3f" %(yp[i][i]))
        print("-----------------------------------------------------")
        sys.stdout.flush()
    return [current_acc, current_loss, run]

    
def savemodel(model,problem):
    if problem.endswith('.h5'):
        filename = problem
    else:
        filename = os.path.join(models_dir, '%s.h5' %problem)
    model.save(filename)
    #W = model.get_weights()
    #print(W)
    #np.savez(filename, weights = W)
    print("\nModel saved successfully on file %s\n" %filename)

    
def loadmodel(problem):
    if problem.endswith('.h5'):
        filename = problem
    else:
        filename = os.path.join(models_dir, '%s.h5' %problem)
    try:
        model = load_model(filename)
        print("\nModel loaded successfully from file %s\n" %filename)
    except:    
        print("\nModel file %s not found!!!\n" %filename)
        model = None
    return model

        
def relu(x):
    z = np.zeros(len(x))
    for i in range(0,len(x)):
        z[i] = max(0,x[i])
    return z

def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def print_solution(model,X):

    np.set_printoptions(precision=3, formatter={'float':lambda x: '%6.3f' %x}, suppress=True)

    print("\n\nModel\n")

    m = model.get_config()
    #print(m)

    input_size = m[0]['config']['batch_input_shape'][1]
    output_size = m[len(m)-1]['config']['units']
    nlayers = len(model.layers)
    
    print("Input size: %d" %input_size)
    print("Output size: %d" %output_size)
    print("Number of layers: %d" %nlayers)

    il = 1    
    for l in model.layers:
        print('Layer %d' %il)
        ml = l.get_config()
        #print(ml)
        try:
            p = ml['batch_input_shape'] # (None, n)
            print("   input shape: %r" %(p[1]))
        except:
            pass
        print("   units: %d" %(ml['units']))
        print("   activation: %s" %(ml['activation']))
        il += 1
        
    print('----\n')

    il = 1
    for l in model.layers:
        print('Layer %d' %il)
        ml = l.get_config()
        och = 'h'
        ich = 'h'
        if il==1:
            ich = 'x'
        if il==nlayers:
            och = 'y'
        print("  %c = %s(W%d%d^T %c + W%d%d)" 
            %(och,ml['activation'],il,1,ich,il,2))
        il += 1
        
    print('----\n')
    
    W = model.get_weights()
    
    print("\nWeights\n")
    il = 1
    for l in model.layers:
        print('Layer %d' %il)
        iw = 1
        for w in l.get_weights():
            print("\nW%d%d = \n%s" %(il,iw,w))
            iw += 1
        il += 1
    
    print('----\n')

    print("\nTest")
    for x in X:
        t = np.transpose(W[0]).dot(x) + W[1]
        f1 = eval(m[0]['config']['activation'])
        f2 = eval(m[1]['config']['activation'])
        h = f1(t)
        #if len(X[0])==2:
        #    h = relu(t)
        #else:
        #    h = sigmoid(t)
        #print(h)
        t = np.transpose(W[2]).dot(h) + W[3]
        y = f2(t)
        strx = "%s" %x
        stry = "%s" %y
        if (len(x)>2):
            strx = '...'
        if (len(y)>2):
            stry = '...'        
        print("    x = %s -> h = %s -> y: %s"  %(strx,h,stry))
    
    print('----\n')
