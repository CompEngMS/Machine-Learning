import sys, os, datetime
import numpy as np
import argparse

import keras
from keras import models, layers, backend
from keras.layers import Dense, Activation, Dropout, Flatten,\
    Conv2D, MaxPooling2D, AveragePooling2D

models_dir = 'models'

def load_data():
    # load data
    (Xtrain,Ytrain), (Xtest, Ytest) = keras.datasets.mnist.load_data()
    # get information
    ninput = Xtrain.shape[0]
    imgsize = (Xtrain.shape[1], Xtrain.shape[2])
    input_shape = (Xtrain.shape[1], Xtrain.shape[2], 1)
    ntest = Xtest.shape[0]
    num_classes = max(Ytrain) + 1
    print("Training input %s" %str(Xtrain.shape))
    print("Training output %s" %str(Ytrain.shape))
    print("Test input %s" %str(Xtest.shape))
    print("Test output %s" %str(Ytest.shape))
    print("Input shape: %s" %str(input_shape))
    print("Number of classes: %d" %num_classes)
    
    # normalize input to [0,1]
    Xtrain = Xtrain / 255.0
    Xtest = Xtest / 255.0
    # reshape input in 4D array
    Xtrain = Xtrain.reshape(ninput,imgsize[0],imgsize[1],1)
    Xtest = Xtest.reshape(ntest,imgsize[0],imgsize[1],1)
    
    # Transform output to one-out-of-n encoding
    Ytrain = keras.utils.to_categorical(Ytrain, num_classes)
    Ytest = keras.utils.to_categorical(Ytest, num_classes)
    
    return [Xtrain,Ytrain,Xtest,Ytest,input_shape,num_classes]


    
def LeNet(input_shape, num_classes):
    
    print('\nLeNet model')
    model = models.Sequential()
    
    print('\tC1: Convolutional 6 kernels 5x5')
    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding="same"))
    print('\tS2: Average Pooling 2x2 stride 2x2')
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    print('\tC3: Convolutional 16 kernels 5x5')
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    print('\tS4: Average Pooling 2x2 stride 2x2')
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    print('\tC5: Convolutional 120 kernels 5x5')
    model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(Flatten())
    print('\tF6: Fully connected, 84 units')
    model.add(Dense(84, activation='tanh'))
    print('\tF7: Fully connected, 10 units')
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = 'adam' #alternative 'SGD'
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    
    return model
        


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
        model = models.load_model(filename)
        print("\nModel loaded successfully from file %s\n" %filename)
    except OSError:    
        print("\nModel file %s not found!!!\n" %filename)
        model = None
    return model

    
### main ###
if __name__ == "__main__":

    problem = 'lenet_mnist'

    parser = argparse.ArgumentParser(description='Keras LeNet example')
    parser.add_argument('-seed', type=int, help='random seed', default=0)
    args = parser.parse_args()

    [Xtrain,Ytrain,Xtest,Ytest,input_shape,num_classes] = load_data()
    
    # Load or create model
    model = loadmodel(problem)
    if model==None:
        model = LeNet(input_shape, num_classes)
    model.summary()
    
    # Set random seed
    if args.seed==0:
        dt = datetime.datetime.now()
        rs = int(dt.strftime("%y%m%d%H%M"))
    else:
        rs = args.seed
    np.random.seed(rs)
    
    print("\nRandom seed %d" %rs)
    
    print("\nTraining ...")
    
     # Train
    try:
        model.fit(Xtrain, Ytrain, batch_size=32, epochs=100)
    except KeyboardInterrupt:
        pass

    print("\n\nEvaluation ...")

    try:
        score = model.evaluate(Xtest, Ytest)
        print("Test loss: %f" %score[0])
        print("Test accuracy: %f" %score[1])
    except KeyboardInterrupt:
        pass

    # Save the model
    savemodel(model,problem)

