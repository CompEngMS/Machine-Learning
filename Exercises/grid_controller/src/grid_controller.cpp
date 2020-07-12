#include "grid_controller.h"
#include <iostream>
#include <sstream>

#define TRUE_TRANSITION_OK  0.7
#define TRUE_TRANSITION_ERR ((1-TRUE_TRANSITION_OK)/2)

#define TRUE_OBSERVATION_OK  0.9
#define TRUE_OBSERVATION_ERR (1-TRUE_OBSERVATION_OK)

#define ESTIMATED_TRANSITION_OK  0.65
#define ESTIMATED_TRANSITION_ERR ((1-ESTIMATED_TRANSITION_OK)/2)

#define ESTIMATED_OBSERVATION_OK  0.94
#define ESTIMATED_OBSERVATION_ERR (1-ESTIMATED_OBSERVATION_OK)



using namespace std;

bool World::load(const char* filename) {
  _map=cv::imread(filename);
  if (_map.rows==0) return false;
  _cell_to_index = cv::Mat(_map.rows, _map.cols, CV_16UC1);
  _n_states = 0;
  for (int r = 0; r<_map.rows; r++)
    for (int c = 0; c<_map.cols; c++) {
      cv::Vec3b& v = _map.at<cv::Vec3b>(r,c);
      if (v[0]==255){
        _cell_to_index.at<short>(r,c) = _n_states;
        _index_to_cell.push_back(std::make_pair(r,c));
        _n_states++;
      } else {
        v=cv::Vec3b(0,0,0);
        _cell_to_index.at<short>(r,c) = -1;
      }
    }
  return true;
}

RCState World::toRCState(int state) {
   std::pair<int, int> state_cell = _index_to_cell[state];
   return RCState(state_cell.first, state_cell.second);
}

int World::toState(RCState state) {
   return _cell_to_index.at<short>(state.r,state.c);
}

bool World::obstacle(int state) {
    return obstacle(toRCState(state));
}

bool World::obstacle(RCState state) {
    cv::Vec3b v = _map.at<cv::Vec3b>(state.r,state.c);
    return (v[0]==0);
}




void LocalizerMap::init() {
  _b.resize(_w._n_states);
  
  float uniform_value   = 1./_b.size();
  std::fill(_b.begin(), _b.end(), uniform_value);
}


void LocalizerMap::showBelief() {
  cv::Mat belief_image(_w._map.rows, _w._map.cols, CV_32FC1);
  belief_image = 1.0f;
  for (size_t s = 0; s<_b.size(); s++) {
    std::pair<int, int> s_cell = _w._index_to_cell[s];
    belief_image.at<float>(s_cell.first, s_cell.second) = 1.0f-_b[s];
  }

  cv::Mat belief_image2;
  cv::resize(belief_image, belief_image2, cv::Size(), 10, 10, cv::INTER_NEAREST);
  cv::imshow("belief", belief_image2);
}

float LocalizerMap::transitionModel(int from, int to, Action action) {
  std::pair<int, int> from_cell = _w._index_to_cell[from];
  std::pair<int, int> to_cell = _w._index_to_cell[to];

  // check if destination cell is valid
  int rt = to_cell.first, ct = to_cell.second;
  if (rt<0 || rt>_w._map.rows-1 || ct<0 || ct>_w._map.cols-1)
    return 0;

  int dr = to_cell.first - from_cell.first ;
  int dc = to_cell.second - from_cell.second;
  if (fabs(dr)>1 || fabs(dc)>1)
    return 0;

  // compute the leading cell, based on the current state and the action
  int r = from_cell.first;
  int c = from_cell.second;
  
  switch (action) {
  case Up: r--; break;
  case Down: r++; break;
  case Left: c--; break;
  case Right: c++; break;
  }
  
  // if the leading cell is invalid, we stay in the same state
  bool invalid_motion = false;
  if (r<0 || r>_w._map.rows-1 || c<0 || c>_w._map.cols-1)
    invalid_motion = true;

  cv::Vec3b v = _w._map.at<cv::Vec3b>(r,c);
   if (v[0]!=255)
     invalid_motion=true;;
 
  if (invalid_motion){
#if 0
    if (dr==0 && dc==0)
      return 1;
    return 0;
#else
    // bounching
    switch (action) {
      case Up:
        if (dr==1 && dc == 0) return ESTIMATED_TRANSITION_OK;
        if (dr==1 && dc == 1) return ESTIMATED_TRANSITION_ERR;
        if (dr==1 && dc == -1) return ESTIMATED_TRANSITION_ERR;
        return 0;
      case Down:
        if (dr==-1 && dc == 0) return ESTIMATED_TRANSITION_OK;
        if (dr==-1 && dc == 1) return ESTIMATED_TRANSITION_ERR;
        if (dr==-1 && dc == -1) return ESTIMATED_TRANSITION_ERR;
        return 0;
      case Left:
        if (dr==0  && dc == 1) return ESTIMATED_TRANSITION_OK;
        if (dr==1  && dc == 1) return ESTIMATED_TRANSITION_ERR;
        if (dr==-1 && dc == 1) return ESTIMATED_TRANSITION_ERR;
        return 0;
      case Right:
        if (dr==0  && dc == -1) return ESTIMATED_TRANSITION_OK;
        if (dr==1  && dc == -1) return ESTIMATED_TRANSITION_ERR;
        if (dr==-1 && dc == -1) return ESTIMATED_TRANSITION_ERR;
        return 0;
#endif
      }  
  }


  // otherwise we compute the transition
  switch (action) {
  case Up: 
    if (dr==-1 && dc == 0) return ESTIMATED_TRANSITION_OK;
    if (dr==-1 && dc == 1) return ESTIMATED_TRANSITION_ERR;
    if (dr==-1 && dc == -1) return ESTIMATED_TRANSITION_ERR;
    return 0;
  case Down: 
    if (dr==1 && dc == 0) return ESTIMATED_TRANSITION_OK;
    if (dr==1 && dc == 1) return ESTIMATED_TRANSITION_ERR;
    if (dr==1 && dc == -1) return ESTIMATED_TRANSITION_ERR;
    return 0;
  case Left: 
    if (dr==0  && dc == -1) return ESTIMATED_TRANSITION_OK;
    if (dr==1  && dc == -1) return ESTIMATED_TRANSITION_ERR;
    if (dr==-1 && dc == -1) return ESTIMATED_TRANSITION_ERR;
    return 0;
  case Right: 
    if (dr==0  && dc == 1) return ESTIMATED_TRANSITION_OK;
    if (dr==1  && dc == 1) return ESTIMATED_TRANSITION_ERR;
    if (dr==-1 && dc == 1) return ESTIMATED_TRANSITION_ERR;
    return 0;

  }
}


float LocalizerMap::observationModel(int state, Observation z) {
  std::pair<int, int> state_cell = _w._index_to_cell[state];
  int r = state_cell.first;
  int c = state_cell.second;

  bool up_occupied= true;
  if (r>0) {
    cv::Vec3b v = _w._map.at<cv::Vec3b>(r-1,c);
    if (v[0]==255)
      up_occupied = false;
  }

  bool down_occupied= true;
  if (r<_w._map.rows-1) {
    cv::Vec3b v = _w._map.at<cv::Vec3b>(r+1,c);
    if (v[0]==255)
      down_occupied = false;
  }


  bool left_occupied= true;
  if (c>0) {
    cv::Vec3b v = _w._map.at<cv::Vec3b>(r,c-1);
    if (v[0]==255)
      left_occupied = false;
  }



  bool right_occupied= true;
  if (c<_w._map.cols-1) {
    cv::Vec3b v = _w._map.at<cv::Vec3b>(r,c+1);
    if (v[0]==255)
      right_occupied = false;
  }

  float cumulative_prob = 1;
  if (up_occupied == (bool)(z & 0x01))
    cumulative_prob *= ESTIMATED_OBSERVATION_OK;
  else
    cumulative_prob *= ESTIMATED_OBSERVATION_ERR;
    

  if (down_occupied == (bool) (z & 0x02))
    cumulative_prob *= ESTIMATED_OBSERVATION_OK;
  else
    cumulative_prob *= ESTIMATED_OBSERVATION_ERR;

  if (left_occupied == (bool) (z & 0x04))
    cumulative_prob *= ESTIMATED_OBSERVATION_OK;
  else
    cumulative_prob *= ESTIMATED_OBSERVATION_ERR;

  if (right_occupied == (bool) (z & 0x08))
    cumulative_prob *= ESTIMATED_OBSERVATION_OK;
  else
    cumulative_prob *= ESTIMATED_OBSERVATION_ERR;
  
  return cumulative_prob;
  
}


void LocalizerMap::predict(Action action) {
  Belief _b2(_b.size());
  std::fill(_b2.begin(), _b2.end(), 0);
  for (size_t from = 0; from<_b.size(); from++) {
    if (_b[from] == 0)
      continue;

    for (size_t to = 0; to<_b.size(); to++) {
      _b2[to] += _b[from] * transitionModel(from, to, action);
    }
  }
  _b = _b2;
}

void LocalizerMap::update(Observation z) {
  double normalizer = 0;
  for (size_t x = 0; x<_b.size(); x++) {
    _b[x] *= observationModel(x, z);
    normalizer += _b[x];
  }
  normalizer = 1./normalizer;
  for (size_t x = 0; x<_b.size(); x++) {
    _b[x] *= normalizer;
  }
  
}


/* Simulator implementation */


float Simulator::transitionModel(int from, int to, Action action) {
  std::pair<int, int> from_cell = _w._index_to_cell[from];
  std::pair<int, int> to_cell = _w._index_to_cell[to];

  // check if destination cell is valid
  int rt = to_cell.first, ct = to_cell.second;
  if (rt<0 || rt>_w._map.rows-1 || ct<0 || ct>_w._map.cols-1)
    return 0;

  // check if destination cell is not adjacent
  int dr = to_cell.first - from_cell.first ;
  int dc = to_cell.second - from_cell.second;
  if (fabs(dr)>1 || fabs(dc)>1)
    return 0;

  // compute the leading cell, based on the current state and the action
  int r = from_cell.first;
  int c = from_cell.second;

  switch (action) {
  case Up: r--; break;
  case Down: r++; break;
  case Left: c--; break;
  case Right: c++; break;
  }


  bool invalid_motion = false;

  cv::Vec3b v = _w._map.at<cv::Vec3b>(r,c);
   if (v[0]!=255)
     invalid_motion=true;

  if (invalid_motion){
    // bounching
    switch (action) {
      case Up:
        if (dr==1 && dc == 0) return TRUE_TRANSITION_OK;
        if (dr==1 && dc == 1) return TRUE_TRANSITION_ERR;
        if (dr==1 && dc == -1) return TRUE_TRANSITION_ERR;
        return 0;
      case Down:
        if (dr==-1 && dc == 0) return TRUE_TRANSITION_OK;
        if (dr==-1 && dc == 1) return TRUE_TRANSITION_ERR;
        if (dr==-1 && dc == -1) return TRUE_TRANSITION_ERR;
        return 0;
      case Left:
        if (dr==0  && dc == +1) return TRUE_TRANSITION_OK;
        if (dr==1  && dc == +1) return TRUE_TRANSITION_ERR;
        if (dr==-1 && dc == +1) return TRUE_TRANSITION_ERR;
        return 0;
      case Right:
        if (dr==0  && dc == -1) return TRUE_TRANSITION_OK;
        if (dr==1  && dc == -1) return TRUE_TRANSITION_ERR;
        if (dr==-1 && dc == -1) return TRUE_TRANSITION_ERR;
        return 0;

      }  
  }


  // otherwise we compute the transition
  switch (action) {
  case Up:
    if (dr==-1 && dc == 0) return TRUE_TRANSITION_OK;
    if (dr==-1 && dc == 1) return TRUE_TRANSITION_ERR;
    if (dr==-1 && dc == -1) return TRUE_TRANSITION_ERR;
    return 0;
  case Down:
    if (dr==1 && dc == 0) return TRUE_TRANSITION_OK;
    if (dr==1 && dc == 1) return TRUE_TRANSITION_ERR;
    if (dr==1 && dc == -1) return TRUE_TRANSITION_ERR;
    return 0;
  case Left:
    if (dr==0  && dc == -1) return TRUE_TRANSITION_OK;
    if (dr==1  && dc == -1) return TRUE_TRANSITION_ERR;
    if (dr==-1 && dc == -1) return TRUE_TRANSITION_ERR;
    return 0;
  case Right:
    if (dr==0  && dc == 1) return TRUE_TRANSITION_OK;
    if (dr==1  && dc == 1) return TRUE_TRANSITION_ERR;
    if (dr==-1 && dc == 1) return TRUE_TRANSITION_ERR;
    return 0;
  }
}


void Simulator::init() {
  // set initial state
  int r=_w._map.rows/2, c=1;
  _current_state = _w._cell_to_index.at<short>(r,c);
}


RCState Simulator::getCurrentState() {
    return _w.toRCState(_current_state);
}


float Simulator::observationModel(int state, Observation z) {
  std::pair<int, int> state_cell = _w._index_to_cell[state];
  int r = state_cell.first;
  int c = state_cell.second;

  bool up_occupied= true;
  if (r>0) {
    cv::Vec3b v = _w._map.at<cv::Vec3b>(r-1,c);
    if (v[0]==255)
      up_occupied = false;
  }

  bool down_occupied= true;
  if (r<_w._map.rows-1) {
    cv::Vec3b v = _w._map.at<cv::Vec3b>(r+1,c);
    if (v[0]==255)
      down_occupied = false;
  }

  bool left_occupied= true;
  if (c>0) {
    cv::Vec3b v = _w._map.at<cv::Vec3b>(r,c-1);
    if (v[0]==255)
      left_occupied = false;
  }

  bool right_occupied= true;
  if (c<_w._map.cols-1) {
    cv::Vec3b v = _w._map.at<cv::Vec3b>(r,c+1);
    if (v[0]==255)
      right_occupied = false;
  }

  float cumulative_prob = 1;
  if (up_occupied == (bool)(z & 0x01))
    cumulative_prob *= TRUE_OBSERVATION_OK;
  else
    cumulative_prob *= TRUE_OBSERVATION_ERR;

  if (down_occupied == (bool) (z & 0x02))
    cumulative_prob *= TRUE_OBSERVATION_OK;
  else
    cumulative_prob *= TRUE_OBSERVATION_ERR;

  if (left_occupied == (bool) (z & 0x04))
    cumulative_prob *= TRUE_OBSERVATION_OK;
  else
    cumulative_prob *= TRUE_OBSERVATION_ERR;

  if (right_occupied == (bool) (z & 0x08))
    cumulative_prob *= TRUE_OBSERVATION_OK;
  else
    cumulative_prob *= TRUE_OBSERVATION_ERR;

  return cumulative_prob;

}

int Simulator::sampleNextState(int from, Action a) {
  float cumval = 0;
  int to = from;
  float v = (rand() / (RAND_MAX + 1.0)); // drand48()
  for (size_t i = 0; i<_w._n_states; i++) {
    cumval+=transitionModel(from, i, a);
    if (cumval>v) {
      to = i; break;
    }
  }
  // cout << "from = " << _w.strstate(from) << " a = " << a << " to = " << _w.strstate(to) << endl;
  return to;
}

Observation Simulator::sampleObservation(int x) {
  float cumval = 0;
  float v = (rand() / (RAND_MAX + 1.0)); // drand48()
  Observation z = 0;
  for (Observation j=0;j<16; j++){
    cumval+=observationModel(x,j);
    if (cumval>v) {
      z = j; break;
    }
  }
  // cout << "Observation: " << z << "  True: " << trueObservation(x) << endl;
  return z;
}


Observation Simulator::trueObservation(int state) {
  std::pair<int, int> state_cell = _w._index_to_cell[state];
  int r = state_cell.first;
  int c = state_cell.second;

  bool up_occupied= true;
  if (r>0) {
    cv::Vec3b v = _w._map.at<cv::Vec3b>(r-1,c);
    if (v[0]==255)
      up_occupied = false;
  }

  bool down_occupied= true;
  if (r<_w._map.rows-1) {
    cv::Vec3b v = _w._map.at<cv::Vec3b>(r+1,c);
    if (v[0]==255)
      down_occupied = false;
  }

  bool left_occupied= true;
  if (c>0) {
    cv::Vec3b v = _w._map.at<cv::Vec3b>(r,c-1);
    if (v[0]==255)
      left_occupied = false;
  }

  bool right_occupied= true;
  if (c<_w._map.cols-1) {
    cv::Vec3b v = _w._map.at<cv::Vec3b>(r,c+1);
    if (v[0]==255)
      right_occupied = false;
  }

  Observation z =
    ((int)up_occupied * 0x01) + ((int)down_occupied * 0x02) + ((int)left_occupied * 0x04) + ((int)right_occupied * 0x08);
  
  return z;
}


Observation Simulator::execute_step(Action a) {

    // sample from the simulator the *true next state*
    _current_state=sampleNextState(_current_state, a);

    // sample from the simulator a feasible observation
    _current_observation=sampleObservation(_current_state);

    // memorize the true observation 
    _true_observation = trueObservation(_current_state);

    return _current_observation;
}


ostream& operator<< (ostream &stream, Observation obs) {
    stream << ((obs&1)>0) << " "
    << ((obs&2)>0) << " "
    << ((obs&4)>0) << " "
    << ((obs&8)>0);
    return stream;
}

ostream& operator<< (ostream &stream, Action a) {
    stream << (a==Up?"U":(a==Down?"D":(a==Left?"L":"R")));
    return stream;
}

ostream& operator<< (std::ostream &stream, RCState x) {
    stream << "(" << x.r << "," << x.c << ")";
    return stream;
}

bool operator< (const RCState &s1, const RCState &s2) {
    return (s2.c>s1.c) || ((s2.c==s1.c) && (s2.r>s1.r));
}

Controller* createController(std::string which, World &w)
{
    if (which=="Right") return new RightController(w);
    if (which=="LeftRight") return new LeftRightController(w);
    if (which=="MDP") return new MDPController(w);
    if (which=="RL") return new RLController(w);
}



float MDPController::reward(RCState s, Action a, RCState s1) {
    if (a==Right && _w.obstacle(RCState(s.r,s.c+1)))
        return -10;
    if (a==Left && _w.obstacle(RCState(s.r,s.c-1)))
        return -10;
    if (s1.c==_w._map.cols-2) // action leading to the last column
        return 1000;
    if (a==Left)
        return -1; 
    else if (a==Right)
        return +1; 
}

float MDPController::transition(RCState s1, RCState s, Action a) {
    int from=_w.toState(s), to=_w.toState(s1);
    return _loc->transitionModel(from,to,a);
}

float MDPController::computeValue(RCState s, Action a) {
    // ∑s' P(s'|s,a) (R(s,a,s')+ γVk-1[s'])

    float sum = 0.f;
    // for each state
    for (int c1=s.c-1; c1<=s.c+1; c1++)
      for (int r1=s.r-1; r1<=s.r+1; r1++) {
          RCState s1(r1,c1);
          sum += transition(s1,s,a) * ( reward(s,a,s1) + (gamma * V.at<float>(r1,c1)) );
      }
    return sum;
}

void MDPController::computePolicy() {

    cout << "Computing the policy (map: " << V.cols << "x" << V.rows << ")  ";
    // iterations
    for (int n=0; n<100; n++) {
      cout << "."; cout.flush();
      // for each state
      for (int c=1; c<V.cols-1; c++)
        for (int r=1; r<V.rows-1; r++) {
            // cout << r << "," << c << endl;
            if (!_w.obstacle(RCState(r,c))) {
                // Vk[s] = max_a ∑s' P(s'|s,a) (R(s,a,s')+ γVk-1[s'])
                float vright = computeValue(RCState(r,c),Right);
                float vleft = computeValue(RCState(r,c),Left);
                V_new.at<float>(r,c) = std::max(vright,vleft);
            }
            else
                V_new.at<float>(r,c) = 0;
        }

      V = V_new;
    }

    // now compute the policy

    // for each state
    for (int c=1; c<V.cols-1; c++)
      for (int r=1; r<V.rows-1; r++) {
        //  π[s] = argmaxa ∑s' P(s'|s,a) (R(s,a,s')+ γVk[s'])
        float vright = computeValue(RCState(r,c),Right);
        float vleft = computeValue(RCState(r,c),Left);
        if (vleft>vright)
            policy.at<unsigned char>(r,c) = 2; // Left
        else
            policy.at<unsigned char>(r,c) = 3; // Right
      }

}

Action MDPController::doAction(RCState state) {
    switch (policy.at<unsigned char>(state.r,state.c)) {
    case 0: return Up;
    case 1: return Down;
    case 2: return Left;
    case 3: return Right;
    }
}


Action RLController::chooseAction(RCState state) {
   
    Action a = Right;
    // choose an action (e-greedy)
    float v1 = (rand() / (RAND_MAX + 1.0)); // drand48()
    float v2 = (rand() / (RAND_MAX + 1.0)); // drand48()
    if (v1<epsilon) { // exploration
        a = (v2<0.5)?Left:Right;
    }
    else { // exploitation
        float ql = Q[make_pair(state,Left)], qr = Q[make_pair(state,Right)];
        a = (ql>qr)?Left:Right;
    }

    return a;
}

Action RLController::doAction(RCState state) {
    this->action = chooseAction(state); this->state = state;
    return action;
}



void RLController::updateQ(RCState succstate) {
    // action: action just executed, state: previous state
    float r = reward(state,action,succstate);
    double alpha = 0.5;
    // Q[s,a] ← Q[s,a] + α(r + γQ[s',a'] - Q[s,a])
    Action action2 = chooseAction(succstate);

    Q[make_pair(state,action)] += alpha * ( r + gamma * Q[make_pair(succstate,action2)] - Q[make_pair(state,action)] );
}

void RLController::computePolicy() {
    // for each state
    for (int c=0; c<_w._map.cols; c++)
      for (int r=0; r<_w._map.rows; r++) {
        float ql = Q[make_pair(RCState(r,c),Left)], qr = Q[make_pair(RCState(r,c),Right)];
        if (ql==0 && qr==0)
            policy.at<unsigned char>(r,c) = 9; // Invalid value
        else if (ql==qr)
            policy.at<unsigned char>(r,c) = 6; // same value
        else
            policy.at<unsigned char>(r,c) = (ql>qr)?2:3; // Left or Right
      }
}
