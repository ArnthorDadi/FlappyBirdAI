from ple.games.flappybird import FlappyBird
from ple import PLE
import random
from Stopwatch import StopWatch

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import os

bird_score = []

class FlappyAgent:
    def __init__(self):
        # TODO: you may need to do some initialization for your agent here
        # State is
        #     (
        #       player_y,                   int 1-15 15 being the highest
        #       next_pipe_top_y,            int 1-15 15 being the highest
        #       next_pipe_dist_to_player,   int 1-15 15 being the longest dist from player
        #       player_vel                  int -10-8 +number means that the player is going up and -numbers down
        #     )
        #                             0's score  1's score
        # self.Q[(state)] = np.array [     0    ,    1    ]
        self.Q = dict()
        # Training policy
        #   np.array
        #     [
        #       chance of 0   inital 50%
        #       chance of 1   inital 50%
        #     ]
        self.PI = dict()

        self.init_q = 0
        self.epsilon = 0
        self.alpha = 0

    def getCell(self, pixel, measurements):
        const = measurements/15
        for cell in range(1, 16):
            if( pixel <= ( cell * const ) ):
                return int(cell)
        print(pixel, measurements)
        print('GET CELL ERROR')
        quit()
    
    def correctStateFormat(self, state):
        if(state['player_vel'] < -8):
            state['player_vel'] = -8
        elif(10 < state['player_vel']):
            state['player_vel'] = 10
        if(288 < state['next_pipe_dist_to_player']):
            state['next_pipe_dist_to_player'] = 288
        return (
            int(self.getCell(state['player_y'], 512)),
            int(self.getCell(state['next_pipe_top_y'], 512)),
            int(self.getCell(state['next_pipe_dist_to_player'], 288)),
            int(state['player_vel'])
        )

    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    
    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        # TODO: learn from the observation
        return

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        #print("state: %s" % state)
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.

        currState = self.correctStateFormat(state)

        if(not currState in self.Q):
            self.Q[currState] = np.array([self.init_q, self.init_q])
        
        best_action_chance = ( 1 - self.epsilon + float(self.epsilon/2) )
        rand_numb = float(random.randint(1, 1000)/1000)
        # exploid
        if(rand_numb <= best_action_chance):
            return np.argmax(self.Q[currState])

        # explore
        return random.randint(0, 1)

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        #print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.
        newState = self.correctStateFormat(state)

        #                        0,        1
        # q[state] = np array [score 0, score 1]
        # argmax(q[state]) = index of higest score
        if(newState not in self.Q):
            return random.randint(0, 1)

        return np.argmax(self.Q[newState])

class OP_MCC_A(FlappyAgent):
    def __init__(self):
        self.curr_episode = []
        self.discountFactor = 0
        super().__init__()
    
    def maxValue(self, valueArray):
        return valueArray[np.argmax(valueArray)]

    def observe(self, s1, a, r, s2, end):
        newState = self.correctStateFormat(s1)
                
        if(end):
            # Go over
            self.curr_episode = [(newState, a, r)] + self.curr_episode

            G = 0
            for (state, action, reward) in self.curr_episode:
                action = int(action)
                G = reward + self.discountFactor * G
                argMax = self.maxValue(self.Q[state])
                newValue = self.Q[state][action] + self.alpha * ( G - argMax )
                self.Q[state][action] = newValue
                
            self.curr_episode = []
            return

        self.curr_episode = [(newState, a, r)] + self.curr_episode

class Q_LEARNING_A(FlappyAgent):
    def __init__(self):
        self.discountFactor = 0
        super().__init__()
    
    def maxValue(self, valueArray):
        #print('Value array: ', valueArray)
        return valueArray[np.argmax(valueArray)]

    def observe(self, s1, a, r, s2, end):
        newState = self.correctStateFormat(s1)
        newStatePrime = self.correctStateFormat(s2)
        Q_SA = self.Q[newState][a]
        if(not newStatePrime in self.Q):
            self.Q[newStatePrime] = np.array([self.init_q, self.init_q])
        Q_SprimeA = self.Q[newStatePrime]
        discounted_Q_SprimeA = self.discountFactor * self.maxValue(Q_SprimeA)
        self.Q[newState][a] = Q_SA + self.alpha * ( r + discounted_Q_SprimeA - Q_SA)

class MC_F_A_A(FlappyAgent):
    def __init__(self):
        self.curr_episode = []
        self.discountFactor = 0
        self.theta = np.array([np.array([0.01,0.01,0.01,0.01]), np.array([0.01,0.01,0.01,0.01])])
        super().__init__()

    def phi(self, state):
        return np.array([state[index] for index in range(0, len(state))])

    def Qsa(self, state, action):
        # [] 
        # 
        return np.dot(self.theta[action], self.phi(state))

    def observe(self, s1, a, r, s2, end):
        newState = self.correctStateFormat(s1)
                
        if(end):
            # Go over
            self.curr_episode = [(newState, a, r)] + self.curr_episode

            G = 0
            for (state, action, reward) in self.curr_episode:
                action = int(action)
                G = reward + self.discountFactor * G
                
                # target = G
                # error = target - Qsa(s, a)
                # theta[a] = theta[a] + alpha * error * phi(s)

                error = G - self.Qsa(state, action)
                right_side = (self.alpha * error) * self.phi(state)
                adding_right_left_sides = np.add( self.theta[action] , right_side )

                self.theta[action] = adding_right_left_sides

            self.curr_episode = []
            return

        self.curr_episode = [(newState, a, r)] + self.curr_episode
    
    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        #print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.
        newState = self.correctStateFormat(state)

        return np.argmax([ self.Qsa(newState, 0), self.Qsa(newState, 1) ])

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        #print("state: %s" % state)
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.

        currState = self.correctStateFormat(state)
        
        best_action_chance = ( 1 - self.epsilon + float(self.epsilon/2) )
        rand_numb = float(random.randint(1, 1000)/1000)
        # exploid
        if(rand_numb <= best_action_chance):
            return np.argmax([ self.Qsa(currState, 0), self.Qsa(currState, 1) ])

        # explore
        return random.randint(0, 1)


def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    # TODO: when training use the following instead:
    # reward_values = agent.reward_values
    
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
            reward_values = reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True 
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        # TODO: for training using agent.training_policy instead
        action = agent.policy(env.game.getGameState())

        # step the environment
        reward = env.act(env.getActionSet()[action])
        print("reward=%d" % reward)

        # TODO: for training let the agent observe the current state transition

        score += reward
        
        # reset the environment if the game is over
        if env.game_over():
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0

def train(nb_episodes, agent):
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        state = env.game.getGameState()
        action = agent.training_policy(state)

        # step the environment
        reward = env.act(env.getActionSet()[action])
        print("reward=%d" % reward)

        # let the agent observe the current state transition
        newState = env.game.getGameState()
        agent.observe(state, action, reward, newState, env.game_over())

        score += reward
        # reset the environment if the game is over
        if env.game_over():
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0