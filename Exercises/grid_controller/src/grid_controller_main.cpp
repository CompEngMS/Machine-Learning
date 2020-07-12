#include "grid_controller.h"
#include <iostream>

using namespace std;

string use =
        "Use: grid_controller <map file name> [options]\n"
        "Options:   -verbose: print additional info\n"
        "Options:   -controller [Right|LeftRight|MDP|RL]: activate controller\n";

        
        
void displayAction(Action a, cv::Mat &act) {
    // show action
    act = cv::Scalar(255,255,255);
    act.at<cv::Vec3b>(2,2)=cv::Vec3b(0, 255, 0);

    if (a==Up) { // up
      act.at<cv::Vec3b>(1,2)=cv::Vec3b(0, 0, 255);
    }
    else if (a==Down) { // down
      act.at<cv::Vec3b>(3,2)=cv::Vec3b(0, 0, 255);
    }
    else if (a==Left) { // left
      act.at<cv::Vec3b>(2,1)=cv::Vec3b(0, 0, 255);
    }
    else if (a==Right) { // right
      act.at<cv::Vec3b>(2,3)=cv::Vec3b(0, 0, 255);
    }
    cv::Mat obs3;
    cv::resize(act, obs3, cv::Size(), 50, 50, cv::INTER_NEAREST);
    cv::imshow("action", obs3);
}


void displayGTObservations(Simulator &sim, Observation current_observation, cv::Mat &obs) {

    // read current state
    std::pair<int, int> current_indices=sim._w._index_to_cell[sim._current_state];
    int r = current_indices.first, c = current_indices.second;

    // draw the world with agent position
    cv::Mat gt=sim._w._map.clone();
    gt.at<cv::Vec3b>(r,c)=cv::Vec3b(0, 255, 0);  // agent position green dot

      // set observation matrix for display current observations
    obs = cv::Scalar(255,255,255);
    obs.at<cv::Vec3b>(2,2)=cv::Vec3b(0, 255, 0);
    
    // draw observations (blue dots) and update values for data collection
    if ((current_observation&1)>0) { // up
      gt.at<cv::Vec3b>(r-1,c)=cv::Vec3b(255, 0, 0);
      obs.at<cv::Vec3b>(1,2)=cv::Vec3b(255, 0, 0);
    }
    if ((current_observation&2)>0) { // down
      gt.at<cv::Vec3b>(r+1,c)=cv::Vec3b(255, 0, 0);
      obs.at<cv::Vec3b>(3,2)=cv::Vec3b(255, 0, 0);
    }
    if ((current_observation&4)>0) { // left
      gt.at<cv::Vec3b>(r,c-1)=cv::Vec3b(255, 0, 0);
      obs.at<cv::Vec3b>(2,1)=cv::Vec3b(255, 0, 0);
    }
    if ((current_observation&8)>0) { // right
      gt.at<cv::Vec3b>(r,c+1)=cv::Vec3b(255, 0, 0);
      obs.at<cv::Vec3b>(2,3)=cv::Vec3b(255, 0, 0);
    }

    // show observations
    cv::Mat obs2;
    cv::resize(obs, obs2, cv::Size(), 100, 100, cv::INTER_NEAREST);
    cv::imshow("observations", obs2);

    // show ground truth and belief
    cv::Mat gt2;
    cv::resize(gt, gt2, cv::Size(), 10, 10, cv::INTER_NEAREST);
    cv::imshow("ground_truth", gt2);

    
}


        
int main(int argc, char** argv) {

  //srand48(0); // random seed

  bool verbose = false, usecontroller=false; string whichcontroller="";

  if (argc<2) {
      cout << use << endl;
      return -1;
  }

  for (int i=2; i<argc; i++) {
    if (string(argv[i])=="-h")
      { cout << use << endl; exit(0); }
    if (string(argv[i])=="-verbose")
        verbose = true;
    if (string(argv[i])=="-controller" && i+1<argc)
      { usecontroller = true; whichcontroller = string(argv[i+1]); }

  }


  // create the world (load an image and make it a map)
  World w;
  if (!w.load(argv[1])) {
      cerr << "Cannot open file " << argv[1] << endl;
      cerr << endl << use << endl;
      exit(-1);
  }

  // create the simulator (evolution not known to the agent)
  Simulator sim(w);

  // create the localizer (evolution estimated by the agent)
  LocalizerMap loc(w);

  // used to collect data for learning
  DataCollector dc;

  cv::Mat obs(5, 5, CV_8UC3);
  cv::Mat act(5, 5, CV_8UC3);


  bool run = true;
  bool pause = usecontroller;

  Controller *controller;

  if (usecontroller) {
      controller = createController(whichcontroller,w);
      if (whichcontroller=="MDP") {
          controller->setLocalizer(&loc);
      }
  }


  sim.init();
  loc.init();
  dc.init();
  if (whichcontroller=="MDP")
    controller->init();
  
  // setup the display
  cv::namedWindow("belief");
  cv::namedWindow("ground_truth");
  cv::namedWindow("observations");
  cv::namedWindow("action");
  cv::namedWindow("policy");

  if (!usecontroller) {
      cout << "Use i j k l keys to control the agent." << endl;
  }
  
  int iteration=0;
  
  
  while (run) {

    cout << endl << "==============================" << endl << "Iteration " << ++iteration << endl;
    
    sim.init();
    loc.init();
    dc.init();
    
    if (usecontroller && whichcontroller!="MDP")
        controller->init();
    
    cv::Mat drawpolicy=w._map.clone();

    if (whichcontroller=="MDP" || whichcontroller=="RL") {

        MDPController *ctrl = dynamic_cast<MDPController*>(controller);

        if (whichcontroller=="RL") {
            RLController *rlctrl = dynamic_cast<RLController*>(controller);
            rlctrl->computePolicy();
        }

        for (int c=1; c<w._map.cols-1; c++)
          for (int r=1; r<w._map.rows-1; r++) {
              if (!w.obstacle(RCState(r,c))) {
                  if (ctrl->policy.at<unsigned char>(r,c)==2) // left
                      drawpolicy.at<cv::Vec3b>(r,c)=cv::Vec3b(0,0,255);  // red dot
                  else if (ctrl->policy.at<unsigned char>(r,c)==3) // right
                      drawpolicy.at<cv::Vec3b>(r,c)=cv::Vec3b(0,255,0);  // green dot
                  else if (ctrl->policy.at<unsigned char>(r,c)==6) // same value
                      drawpolicy.at<cv::Vec3b>(r,c)=cv::Vec3b(0,255,200);  // yellow dot
              }
          }

    }


    Observation current_observation = 0;
    double cumrew = 0.0;
    double cumgamma = 1.0;
    
    while (run) { // episode

      // read current state
      std::pair<int, int> current_indices=sim._w._index_to_cell[sim._current_state];
      int r = current_indices.first, c = current_indices.second;


      // read true observation for data collection
      if ((sim._true_observation&1)>0) dc.nobstacles[1]++;
      if ((sim._true_observation&2)>0) dc.nobstacles[2]++;
      if ((sim._true_observation&4)>0) dc.nobstacles[4]++;
      if ((sim._true_observation&8)>0) dc.nobstacles[8]++;
      if ((current_observation&1)>0) { // up
        if ((sim._true_observation&1)>0) dc.nobservations[1]++;
      }
      if ((current_observation&2)>0) { // down
        if ((sim._true_observation&2)>0) dc.nobservations[2]++;
      }
      if ((current_observation&4)>0) { // left
        if ((sim._true_observation&4)>0) dc.nobservations[4]++;
      }
      if ((current_observation&8)>0) { // right
        if ((sim._true_observation&8)>0) dc.nobservations[8]++;
      }

      displayGTObservations(sim,current_observation,obs);
      
      loc.showBelief();

      if (whichcontroller=="MDP" || whichcontroller=="RL") {
          // show the policy
          cv::Mat p2;
          cv::resize(drawpolicy, p2, cv::Size(), 10, 10, cv::INTER_NEAREST);
          cv::imshow("policy", p2);
      }

      if (verbose) {
          // print information
          cout << "Current location: " << current_indices.first << " " <<  current_indices.second << endl;
          cout << "Current observation: " << current_observation << endl;
      }

      

      Action a=Right;

      int waittime = 0; // manual input
      if (usecontroller) waittime = 10;
      if (whichcontroller=="RL") waittime = 1;

      char k = cv::waitKey(waittime);

      // manual input
      // read a key to get the control (i=up, k=down, j=left, l=right)
      switch(k) {
        case 'i': a=Up; break;
        case 'k': a=Down; break;
        case 'j': a=Left; break;
        case 'l': a=Right; break;
        case ' ': pause = !pause; break;
        case 27: run=false;
        default: ;
      }


      // pause/unpause the run
      while (pause && run) {
        char k = cv::waitKey(100);
        if (k==' ') pause=false;
        if (k==27) run=false;
      }

      RCState current_state = sim.getCurrentState();

      // action chosen by the controller
      if (usecontroller)
        a = controller->doAction(current_state);
  
        
      displayAction(a,act);
  
      // remember previous state
      RCState previous_state = current_state;

      /*** EXECUTE THE ACTION ***/

      // update the simulator status
      current_observation = sim.execute_step(a);

      // get the new current state
      current_state = sim.getCurrentState();

      if (whichcontroller=="RL") {
          RLController *ctrl = dynamic_cast<RLController*>(controller);
          ctrl->updateQ(current_state);
      }
      if (whichcontroller=="MDP" || whichcontroller=="RL") {    
          MDPController *ctrl = dynamic_cast<MDPController*>(controller);
          cumrew += cumgamma * ctrl->reward(previous_state, a, current_state);
          cumgamma *= ctrl->gamma;
      }

      // encode the transition in an integer
      int encode = (current_state.r - previous_state.r) + (current_state.c - previous_state.c)*4;

      // update the statistics for data collection
      dc.action_count[a]++;
      dc.action_motion_count[make_pair(a,encode)]++;
      if (a==Right && w.obstacle(RCState(previous_state.r,previous_state.c+1))) {
          dc.action_hit_count[a]++;
      }
      if (a==Left && w.obstacle(RCState(previous_state.r,previous_state.c-1))) {
          dc.action_hit_count[a]++;
      }

      if (verbose) {
          // print information
          cout << "Action: " << a << endl;
          cout << "Transition: " << previous_state << " -> " << current_state << endl;
          cout << "Encode: " << encode << endl;
      }

      // update the filter status
      loc.predict(a);
      loc.update(current_observation);
      
      if (usecontroller) {
      // check if agent reached right-end side of the worlds and terminate the run
        if (sim.getCurrentState().c==sim._w._map.cols-2) {
          break;
        }
      }

    } // while episode


    cout << "N. of R actions = " << dc.action_count[Right] << endl;
    cout << "N. of R actions with hit = " << dc.action_hit_count[Right] << endl;

    cout << "N. of L actions = " << dc.action_count[Left] << endl;
    cout << "N. of L actions with hit = " << dc.action_hit_count[Left] << endl;

    cout << "N. of R actions -> c+1 = " << dc.action_motion_count[make_pair(Right,4)] << endl;
    cout << "N. of R actions -> c-1 = " << dc.action_motion_count[make_pair(Right,-4)] << endl;


    cout << "N. of right obstacles = "  << dc.nobstacles[8] << endl;
    cout << "N. of right obstacles AND right observations = " << dc.nobservations[8] << endl;


    double p = (double) dc.action_motion_count[make_pair(Right,4)] / ( dc.action_count[Right] - dc.action_hit_count[Right]); 
    cout << "Transition probability P(x_t+1 = c+1| x_t = c & not obstacle_right) = " << p << endl;

    p = (double) dc.action_motion_count[make_pair(Right,-4)] / dc.action_hit_count[Right]; 

    cout << "Transition probability P(x_t+1 = c-1| x_t = c & obstacle_right) = " << p << endl;
    
    double pz = (double)dc.nobservations[8] / dc.nobstacles[8];

    cout << "Observation probability P(z:t = right | obstacle_right) = " << pz << endl;

    
    
    if (usecontroller && whichcontroller=="MDP") {
      MDPController *ctrl = dynamic_cast<MDPController*>(controller);
      cout << "MDP - cumulative reward = " << cumrew << endl;
    }

    if (usecontroller && whichcontroller=="RL") {
      RLController *ctrl = dynamic_cast<RLController*>(controller);
      cout << "RL - epsilon = " << ctrl->epsilon << endl;
      cout << "RL - cumulative reward = " << cumrew << endl;
      ctrl->epsilon -= ctrl->epsilon/100.0; 
    }

    if (usecontroller && whichcontroller!="RL") {
      cout << endl << "Press a key to restart or 'Esc' to exit." << endl;
      char k = cv::waitKey(0);
      run = (k!=27); // esc
    }
    
  } // while run
  
}

