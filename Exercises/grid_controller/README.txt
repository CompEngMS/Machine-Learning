This folder contains
- an implementation of a grid-based controller for a mobile robot with 4 bumpers

requires:
- opencv

To compile:

cd grid_controller
mkdir build
cd build
cmake ..
make

binaries are in bin and 

To run:

$ grid_controller <map file name> [options]
Options:   -verbose: print additional info
Options:   -controller [Right|LeftRight|MDP|RL]: activate controller

        

Example 1:

  $ bin/grid_controller map_linear.png

  Drive the robot with i,j,k,l keys and see the belief update

Example 2:

  $ bin/grid_controller map_linear.png -controller [controller_name]

  Available controllers: 
    - Right: always choose Right action
    - LeftRight: random choice between Left and Right
    - MDP: compute policy with Value iteration and then executes it
    - RL: compute policy with Reinforcement Learning 

  Goal of the agent is to reach the right end of the world without hitting obstacles.

  User control: SPACE key to pause, ESC to quit, any key to start/restart the run.

  Number of actions, obstacles hit, estimated transition and observation 
  probabilities, and reward are printed at the end of each run.

