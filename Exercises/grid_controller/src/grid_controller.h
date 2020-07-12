#include "opencv2/opencv.hpp"
#include <vector>
#include <ostream>


// Set of actions
enum Action {Up, Down, Left, Right};

// Observations
typedef char Observation;  // UDLR bit

// State representation
typedef int State;  // integer representation of map cells

struct RCState {  // (row,column) representation of the states
    int r,c;
    RCState() { r=0; c=0; }
    RCState(int _r, int _c) { r=_r; c=_c; }
};


// stream operators
std::ostream& operator<< (std::ostream &stream, Observation obs);
std::ostream& operator<< (std::ostream &stream, Action a);
std::ostream& operator<< (std::ostream &stream, RCState x);
bool operator< (const RCState &s1, const RCState &s2);


typedef std::vector <float> Belief;


/*************** World: environment management ************************/

class World {
public:
    // loads a map from an image. white: free space, black obstacles.
    bool load(const char* filename);

    RCState toRCState(int state); // converts a state in RCState notation
    int toState(RCState state); // converts a RCstate in int notation

    bool obstacle(int state);
    bool obstacle(RCState state);

    /************** member variables ************************/
    cv::Mat _map; // map of the world
    cv::Mat _cell_to_index; // cell to index data structure, to look up a state at given coordinates
    std::vector< std::pair<int, int> > _index_to_cell; // looks up the coordinates given the state

    int _n_states; // total number of states

};



/************** Localizer: belief of the agent *****************/

class LocalizerMap {
 public:
  
  LocalizerMap(World &w) {
      _w=w;
  }

  // initializes the filter with a uniform prior
  void init();

  // display the current belief map
  void showBelief();

  // implements the AGENT'S BELIEF of the transition and observation models
  float transitionModel(int from, int to, Action action);
  float observationModel(int state, Observation o);

  // implements the prediction step of the filter (integrates the action in the belief)
  void predict(Action action);

  // implements the update step of the filter (integrates the observation in the belief)
  void update(Observation z);

  World _w;
  Belief _b;
};



/****** Simulator: controls the evolution, not known to the agent ********/

class Simulator {

public:
    Simulator(World &w) {
        _w=w;
    }

    void init();

    RCState getCurrentState(); // returns current_state in RCState notation

    // executes one step of simulation
    Observation execute_step(Action a);



    // implements the TRUE transition and observation models
    float transitionModel(int from, int to, Action action);
    float observationModel(int state, Observation o);

    // draws a next state from the previous one, given the action a
    int sampleNextState(int from, Action a);

    // draws an observation, given a state
    Observation sampleObservation(int x);

    // returns the true observation in the state
    Observation trueObservation(int state);

    World _w;

    // current real state (not known to the agent)
    State _current_state;

    // current observation according to the sensor model (known to the agent)
    Observation _current_observation;

    // true observation from a perfect sensor (not known to the agent)
    Observation _true_observation;

};



class Controller {
protected:
    World _w;
    LocalizerMap* _loc;

public:
  Controller(World w) : _w(w) {}

  virtual Action doAction(RCState state) = 0;

  virtual void setLocalizer(LocalizerMap *loc) {
      _loc = loc;
  }

  virtual void init() {}
};

Controller* createController(std::string which, World &w);

class RightController : public Controller  {
public:
  RightController(World w) : Controller(w) {}

  virtual Action doAction(RCState state) {
    return Right;
  }
};

class LeftRightController : public Controller  {
public:
  LeftRightController(World w) : Controller(w)  {}

  virtual Action doAction(RCState state) {
      float v = (rand() / (RAND_MAX + 1.0)); // drand48();
      if (v>0.5)
        return Right;
      else
        return Left;
  }
};


class MDPController : public Controller {
protected:

    cv::Mat V, V_new; // value function

    float transition(RCState s1, RCState s, Action a);
    
    float computeValue(RCState s, Action a);
    

public:

  float gamma;

  cv::Mat policy; // policy

  MDPController(World w) : Controller(w)  {
      gamma = 0.9;
      V.create(_w._map.rows,_w._map.cols,CV_32FC1);
      V_new = V.clone();
      policy.create(_w._map.rows,_w._map.cols,CV_8UC1);
  }

  virtual void init() {
      computePolicy();
  }

  float reward(RCState s, Action a, RCState s1);
  
  virtual Action doAction(RCState state);

  void computePolicy();

};



class RLController : public MDPController {
protected:

  std::map < std::pair<RCState, Action>, float > Q; // Q function
  RCState state; // current state
  Action action; // chosen action at current state
  
public:

  RLController(World w) : MDPController(w), epsilon(0.5)  {}

  virtual void init() {}

  virtual Action doAction(RCState state); // set the chosen action

  Action chooseAction(RCState state);   // choose an action

  void updateQ(RCState succstate);  // update the Q function

  void computePolicy(); // compute the policy

  double epsilon; // exploration limit

};





class DataCollector {
public:  
  std::map< Action, int >  action_count;
  std::map< std::pair<Action,int>, int >  action_motion_count;
  std::map< Action, int >  action_hit_count;
  std::map< int, int > nobstacles, nobservations;

  DataCollector() {
  }

  void init() {
    action_count.clear();
    action_motion_count.clear();
    action_hit_count.clear();
    nobstacles.clear(); nobservations.clear();
  }

};


