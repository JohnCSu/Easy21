from numpy.lib.utils import safe_eval
from Easy21 import Easy21
import random
import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

class MonteCarlo_Control():
    def __init__(self,eps = 0.01,num_actions =2, N0 = 100):
        self.N0 = N0
        self.Q = dict()
        self.N = dict()
        self.Q_visited = dict()
        
        self.num_actions = num_actions
        self.alpha = 0
        self.eps = 1
        # self.probs = [self.eps/self.num_actions + 1-self.eps, self.eps/2  ]
    def eps_greedy(self,s):
        # Returns the action idx 'hit' : 0 ,'stick : 1
        if s not in self.Q_visited:
            self.Q[s] =  [0,0]
            self.N[s] =  [0,0]
            self.Q_visited[s] = 0
        
        
        self.eps =  self.N0/(self.N0 + self.Q_visited[s])
        #set argmax
        if self.Q[s][0] >= self.Q[s][1] :
            a_max,a_min = 0,1
        else:
            a_max,a_min = 1,0


        if random.random() < self.eps/self.num_actions + 1-self.eps:
            return np.argmax(self.Q[s])
        else:
            return np.argmin(self.Q[s])

    def generate_Episode(self, game = Easy21() ):
        game.new_Game()
        
        # Generate a list of tuples ( state, reward)
        episode = []
        state_old, r = game.state
        action_old = self.eps_greedy(state_old)
        while(r is None):
            state,r = game.step(state_old, action_old)

            action = self.eps_greedy(state)
            if r is None:
                reward = 0
            else:
                reward = r

            episode.append( (state_old,action_old,reward) )
            state_old, action_old = state, action
        return episode

    def evaluate(self,episode):
        states ,actions, rewards = list(zip(*episode))
        tot_rewards = list(accumulate(rewards[::-1]))[::-1]
        for state,action,reward in zip(states,actions,tot_rewards):
            self.N[state][action] += 1
            alpha = 1/self.N[state][action]
            self.Q[state][action] += alpha*(reward - self.Q[state][action] )
            self.Q_visited[state] += 1
        
        

    def control(self):
        episode = self.generate_Episode(game = Easy21())
        self.evaluate(episode)
        # print(episode)
        return episode[-1][-1] > 0

    def playGame(self,n = 100):
        wins = 0.0
        
        for i in range(n):
            episode =self.generate_Episode(game= Easy21())
            wins += episode[-1][-1] > 0
        print(f'Winner Prob = {wins/n * 100} %')
    
    def plot_Q(self):
        points = (zip(*self.Q.keys()))


        points = np.array( [ list(elem) for elem in points]  ).T
        p_range = range(1,22)
        d_range = range(1,11)
        X,Y = np.meshgrid(p_range,d_range) 
        print(np.shape(points), len([max(self.Q[key]) for key in self.Q.keys() ]))
        Z = griddata(points, [max(self.Q[key]) for key in self.Q.keys() ], (X,Y) )
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X,Y,Z)
        plt.show()

if __name__ == "__main__":
    random.seed(a = 42)
    MC = MonteCarlo_Control()
    wins = 0
    num_ep = 500000
    for i in range(num_ep):
        wins += MC.control()
    print(f'Winner Prob = {wins/num_ep * 100} %')
    #MC.playGame(n=10000)py
    MC.plot_Q()