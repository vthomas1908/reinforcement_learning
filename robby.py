


import math
import sys
import random

#global values: these can be changed for different 
#experiments
#Number of episodes
n = 5000

#number of steps in each episode
m = 200

#learning rate
l_rate = .2

#epsilon starting point
e = 1

#minimum epsilon
e_min = .1

#epsilon decrease by this amount
#each 50 episodes until e = e_min
e_down = .01 

#gamma - discount rate
g = .9

#rewards/penalties
crash = -5
pu_empty = -1
move = 0 
pu = 10
move_zero = -5

#class to work in Robby Robot's world
class World(object):
  def __init__(self):
    #local copies of the global vars
    self.n = n
    self.m = m
    self.l_rate = l_rate
    self.e = e
    self.e_min = e_min
    self.e_down = e_down
    self.g = g
    self.rew_crash = crash
    self.rew_pu_empty = pu_empty
    self.rew_move = move
    self.rew_pu = pu
    self.plot_reward = []

    #there are 3^5 possible states
    #H, N, E, S, W can each be can, empty, or wall
    self.num_states = math.pow(3, 5)
    self.num_sensors = 5

    #assign values to positions/state
    #this will be used to find index into q_matrix
    #H, N, E, S, W will each represent a power (3^x)
    #where can, empty, wall represent one of those 3
    self.here = 4
    self.north = 3
    self.east = 2
    self.south = 1
    self.west = 0
    self.wall = 2
    self.can = 1
    self.empty = 0

    #actions
    self.num_actions = 5
    self.move_north = 0
    self.move_east = 1
    self.move_south = 2
    self.move_west = 3
    self.pick_up = 4

    #the grid space to be filled (some spaces have cans)
    self.world = []
    self.world_size = 10
    for i in range(self.world_size):
      self.world.append([0] * self.world_size)

    #robby's coordinates for position in the world
    self.robby_i = 0
    self.robby_j = 0

    #the q-matrix to be filled
    self.q_matrix = []

    #testing vars
    self.test_mean_total_rew = 0
    self.test_standard_dev = 0

  def __q_matrix_init(self):
    for i in range(int(self.num_states)):
      self.q_matrix.append([0] * self.num_sensors)

  def __world_init(self):

    #fill the world spaces (each space is either can or not)
    for i in range(self.world_size):
      self.world[i] = [0] * self.world_size
      for j in range(self.world_size):
        if random.randint(0, 1) > 0:
          self.world[i][j] += 1

    #get robby's position in the world
    self.robby_i = random.randint(0, self.world_size - 1)
    self.robby_j = random.randint(0, self.world_size - 1)

  def __get_state(self, i, j):
    state_h = state_n = state_e = state_s = state_w = 0

    #check if this space has can
    if self.world[i][j]:
      state_h = self.can
    else:
      state_h = self.empty

    #check if north has can or is wall (north is i-1)
    if i - 1 < 0:
      state_n = self.wall
    elif self.world[i - 1][j]:
      state_n = self.can
    else:
      state_n = self.empty

    #check if east has can or is wall (east is j+1)
    if j + 1 > self.world_size - 1:
      state_e = self.wall
    elif self.world[i][j + 1]:
      state_e = self.can
    else:
      state_e = self.empty

    #check if south has can or is wall (south is i+1)
    if i + 1 > self.world_size - 1:
      state_s = self.wall
    elif self.world[i + 1][j]:
      state_s = self.can
    else:
      state_s = self.empty

    #check if west has can or is wall (west is j-1)
    if j - 1 < 0:
      state_w = self.wall
    elif self.world[i][j - 1]:
      state_w = self.can
    else:
      state_w = self.empty

    #index for state is just a base 3 calculation
    temp = (state_w * math.pow(3, self.west)) \
           + (state_s * math.pow(3, self.south)) \
           + (state_e * math.pow(3, self.east)) \
           + (state_n * math.pow(3, self.north)) \
           + (state_h * math.pow(3, self.here))

    return temp


  def __choose_action(self, idx):
    #return the right action
    action = 0

    #with probability of 1-e, choose action to  max Q(s,a) 
    if round(random.random(), 2) <= (1 - self.e):
      action_vals = []

      #get a list of q_matrix vals for current state
      for a in range(self.num_actions):
        action_vals.append(self.q_matrix[idx][a])

      #pick the largest (if all same, pick random)
      max_val = action_vals[0]
      action = 0
      total = 0

      for a in range(1, self.num_actions):
        total += action_vals[a]

        if action_vals[a] > max_val:
          max_val = action_vals[a]
          action = a

        #in case of tie, break it
        elif action_vals[a] == max_val:
          x = random.randint(0, 1)

          if x:
            max_val = action_vals[a]
            action = a

      #if all same val, pick random action
      if (total/self.num_actions) == total:
        action = random.randint(0, self.num_actions - 1)

    #with probability of e, choose action randomly
    else:
      action = random.randint(0, self.num_actions - 1)

    return action        



  def __action(self, action):
    rew = 0

    #do the action
    #reward ++++ if pick up can
    #reward ---- if crash into wall (move back to old space)
    #            if pick up in empty square
    #            each move (only for one experiment, otherwise
    #                       penalty for move is just 0)
    rew += self.rew_move

    #uncomment this section for testing punishment
    #for landing in an empty square
#    if self.world[self.robby_i][self.robby_j] == 0:
#      rew += -10

    if action == self.move_north:
      if (self.robby_i - 1) < 0:
        rew += self.rew_crash
      else:
        self.robby_i -= 1

    if action == self.move_east:
      if (self.robby_j + 1) > (self.world_size - 1):
        rew += self.rew_crash
      else:
        self.robby_j += 1

    if action == self.move_south:
      if (self.robby_i + 1) > (self.world_size - 1):
        rew += self.rew_crash
      else:
        self.robby_i += 1

    if action == self.move_west:
      if (self.robby_j - 1) < 0:
        rew += self.rew_crash
      else:
        self.robby_j -= 1

    if action == self.pick_up:
      if self.world[self.robby_i][self.robby_j]:
        self.world[self.robby_i][self.robby_j] = 0
        rew += self.rew_pu
      else:
        rew += self.rew_pu_empty

    return rew


  def __update_q_matrix(self, current, action, new, reward):
    #new q val for this state is
    #current q val + learn_rate * (reward +
    #gamma * (next q val for action that maxes it) - 
    #current q val)

    #vars
    q_cur = self.q_matrix[current][action]
    q_next = 0

    #find q_next with action to max the value
    max_val = self.q_matrix[new][0]
    act = 0
    total = 0

    for a in range(1, self.num_actions):
      total += self.q_matrix[new][a]

      if self.q_matrix[new][a] > max_val:
        max_val = self.q_matrix[new][a]
        act = a

    #so the max q_value for new state is:
    q_next = self.q_matrix[new][act]

    #update the q_value for this space
    temp = q_next * self.g
    temp += reward
    temp -= q_cur
    temp *= self.l_rate
    q_cur += temp

    self.q_matrix[current][action] = q_cur


  def __avg_rew(self, rew_list):
    val = 0

    for i in rew_list:
      val += i

    return val/len(rew_list)

  def __standard_dev(self, rew_list):

    #first need to find variance
    variance = 0

    for i in rew_list:
      variance += math.pow((i - self.test_mean_total_rew), 2)

    variance = variance/len(rew_list)

    #standard deviation is the square root of the variance
    return math.sqrt(variance)


  def learn(self):
    #initialize the q_matrix to all zeros (only done once)
    self.__q_matrix_init()

    #tracker to decrement epsilon (dec every 50 eps)
    e_tracker = 50
    e_cnt = 0

    #tracker to plot training reward point (plot every 100 eps)
    plot_tracker = 100
    plot_cnt = 0
    plot_idx = 0
    plot_flag = 0

    for episode in range(n):

      #initialize the world (random cans/robby placement)
      self.__world_init()

      #dec epsilon if above the set min value
      #only when the counter is at the tracker value
      #on dec, reset counter
      if self.e > self.e_min:
        e_cnt += 1
        if e_cnt == e_tracker:
          e_cnt = 0
          self.e -= self.e_down

      #every 100 episodes, plot the reward for the episode
      plot_cnt += 1
      if plot_cnt == plot_tracker:
        plot_cnt = 0
        plot_flag = 1
        self.plot_reward.append(0)

      for step in range(self.m):

        #observe robby's current state
        cur_state = \
            int(self.__get_state(self.robby_i, self.robby_j))

        #choose an action
        action = self.__choose_action(cur_state) 

        #get the reward/punishment for this move
        rew = self.__action(action)

        #plot
        if plot_flag:
          self.plot_reward[plot_idx] += rew

        #observe robby's new state
        new_state = \
            int(self.__get_state(self.robby_i, self.robby_j))

        #update the q_matrix
        self.__update_q_matrix(cur_state, action, new_state, rew)

      #done with episode
      #make sure plot flag turned off, inc list idx
      if plot_flag:
        plot_flag = 0
        plot_idx += 1



  def test(self):

    #set epsilon to zero --> always pick the max action
    self.e = 0

    #set up a reward holder
    rewards = []

    for episode in range(n):

      #initialize the world (random cans/robby placement)
      self.__world_init()

      #initialize reward for the episode
      rewards.append(0)
      for step in range(self.m):

        #observe robby's current state
        cur_state = \
            int(self.__get_state(self.robby_i, self.robby_j))

        #choose an action
        action = self.__choose_action(cur_state)

        #perform the action and receive the reward/punishment
        rew = self.__action(action)
        rewards[episode] += rew

    #done with episodes
    #get the mean sum of rewards per episode
    self.test_mean_total_rew = self.__avg_rew(rewards)

    #get the standard deviation for the test rewards/episode
    self.test_standard_dev = self.__standard_dev(rewards)


  def print_data(self):
    print("Episodes: ", self.n)
    print("Steps: ", self.m)
    print("Learning Rate: ", self.l_rate)
    print("Discount Rate: ", self.g)
    print("Epsilon: ", self.e, " e min: ", self.e_min, end=" ")
    print("e reduce by: ", self.e_down)
    print("Training_Plot ", self.plot_reward)
    print("\nTest_Average: ", self.test_mean_total_rew)
    print("\nTest_Standard_Deviation: ", self.test_standard_dev)




#main
robby_run = World()
robby_run.learn()
robby_run.test()
robby_run.print_data()
