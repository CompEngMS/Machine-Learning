======================================================================
Keras/TF ANN examples
Requires Python3, Tensorflow and Keras 
    https://www.tensorflow.org/
    https://keras.io/

Luca Iocchi 2018
======================================================================

usage: keras_ex1.py [-h] [-hidden HIDDEN] [-model MODEL] [-seed SEED]
                    [-niter NITER] [-lrate LRATE]
                    problem

Keras/TF ANN examples

positional arguments:
  problem         problem (e.g., xor, nl6, noisyxor, binaryid)

optional arguments:
  -h, --help      show this help message and exit
  -hidden HIDDEN  hidden units
  -model MODEL    model filename
  -seed SEED      random seed
  -niter NITER    iterations
  -lrate LRATE    learning rate



To quit an example, press CTRL+C on keyboard, current model is saved in
the models folder with the following file name structure:
<problem>_<random seed>_<accuracy>_<loss>.h5
(accuracy in percentage units, loss in 1e-3 units)


How to:

1) Creating and training new models

    python3 keras_ex1.py <problem> [-hidden <hidden units>] [-seed <random seed>] [-lrate <learning rate>] [-niter <iterations>]

If number of hidden units is not given, a default for each problem 
is used: 2 for xor, nl6 and noisyxor, 8 for binaryid.

If random seed is not given, it will generated from current time:
<year><month><day><hour><minute>

If learning rate is not given, it is initialized as 1e-3 when creating
a new model and remains unchanged when continuing training.

If number of iterations is specified, training terminates after such number of iterations, otherwise it will continue until you press CTRL+C.


Examples:

    python3 keras_ex1.py xor -seed 1811231813

    python3 keras_ex1.py nl6 -seed 1811231848

    python3 keras_ex1.py noisyxor -seed 1811241657

    python3 keras_ex1.py binaryid -seed 1811241431 -lrate 0.1

    python3 keras_ex1.py binaryid -seed 1811241431 -hidden 8 -lrate 0.1

Notes: 
- with different seeds the convergence is not guaranteed or can take much longer time
- binaryid takes a few minutes to converge
- interesting case: nl6, seed=1811251356, lrate 1e-4
    
2) Continue training from a saved model

    python3 keras_ex1.py <problem> -model <modelfile> [-lrate <learning rate>] [-niter <iterations>]

If learning rate is specified, this new value will be used in this
training session and will be stored in the model, otherwise the value
stored in the model is used.

If number of iterations is specified, training terminates after such number of iterations, otherwise it will continue until you press CTRL+C.


3) Print a saved model

    python3 keras_ex1.py <problem> -model <modelfile> -niter 0

This command will only display the model stored in the modelfile.

