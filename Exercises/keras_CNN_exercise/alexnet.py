import os
import numpy as np

import keras
from keras import models, layers, backend
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

import tflearn.datasets.oxflower17 as oxflower17

models_dir = 'models'

def load_data():
    Xtrain, Ytrain = oxflower17.load_data(one_hot=True)
    
    input_shape = (Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3])     # (224,224,3)
    num_classes = Ytrain.shape[1]  # 17
    print("Training input %s" %str(Xtrain.shape))
    print("Training output %s" %str(Ytrain.shape))
    #print("Test input %s" %str(Xtest.shape))
    #print("Test output %s" %str(Ytest.shape))
    print("Input shape: %s" %str(input_shape))
    print("Number of classes: %d" %num_classes)

    return [Xtrain,Ytrain,input_shape,num_classes] 



def AlexNet(input_shape, num_classes):
    # Some details in https://www.learnopencv.com/understanding-alexnet/

    model = Sequential()

    # C1 Convolutional Layer 
    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11),\
     strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # C2 Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C3 Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C4 Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C5 Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Flatten
    model.add(Flatten())

    # D1 Dense Layer
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # D2 Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # D3 Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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

    problem = 'alexnet_oxflower17'

    #np.random.seed(20181205)

    # Get Data
    [Xtrain,Ytrain,input_shape,num_classes] = load_data()
    
    # Load or create model
    model = loadmodel(problem)
    if model==None:
        model = AlexNet(input_shape, num_classes)

    # Summary
    model.summary()

    # Train
    try:
        model.fit(Xtrain, Ytrain, batch_size=64, epochs=100, verbose=1, \
        validation_split=0.2, shuffle=True)
    except KeyboardInterrupt:
        pass
      
    # Save the model
    savemodel(model,problem)

