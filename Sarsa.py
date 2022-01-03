from numpy.lib.utils import safe_eval
from Easy21 import Easy21
import random
import numpy as np
from itertools import accumulate
from Monte_Carlo import MonteCarlo_Control
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pickle
import os
class Sarsa_Control():
    def __init__(self,eps = 0.01,num_actions =2, N0 = 100,lam = 1,gamma = 1):
        self.gamma = gamma
        self.lam = lam
        self.N0 = N0
        self.Q = dict()
        self.N = dict()
        self.Q_visited = dict()
        
        self.num_actions = num_actions
        self.alpha = 0
        self.eps = 1
        

         
        self.init_tables()
        
    def init_tables(self):
        s = [(p,d) for p in range(1,22) for d in range(1,11)]
        self.Q_visited = { state:0 for state in s   }
        self.Q = { state:[0,0] for state in s   }
        self.N = { state:[0,0] for state in s   }
        self.E = { state:[0,0] for state in s   }

        del s
        


    def eps_greedy(self,s):
        # Returns the action idx 'hit' : 0 ,'stick : 1
        self.eps =  self.N0/(self.N0 + self.Q_visited[s])

        if random.random() < self.eps/self.num_actions + 1-self.eps:
            return np.argmax(self.Q[s])
        else:
            return np.argmin(self.Q[s])

    def TD_learn(self, game = Easy21(),lam = 1,n=1 ):
        self.lam = lam
        
        for num_episodes in range(n):
            self.E = { s:[0,0] for s in self.E.keys()}
            game.new_Game()
            state, r = game.state
            action = self.eps_greedy(state)
            stop_ep = False
            while(r is None):
                state_next,r = game.step(state,action)
                
                
                if r is None:
                    action_next = self.eps_greedy(state_next)
                    r = 0
                    self.Q_visited[state] += 1
                    self.E[state][action] += 1
                    
                    self.N[state][action] += 1
                    self.alpha = 1/self.N[state][action]

                    delta = r + self.gamma*self.Q[state_next][action_next] -  self.Q[state][action] 
                    self.E[state][action] += 1
                    self.Q[state][action] += self.alpha*delta*self.E[state][action]
                    
                    self.E = { self.gamma*self.lam*es[a] for es in self.E for a in range(2) } ##############################

                    state,action = state_next,action_next
                else:
                    break
  
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

    filename = 'MC_Q_star.pkl'
    if os.path.exists('MC_Q_star.pkl' ):
        with open(filename, 'rb') as f:
            Q_star = pickle.load(f)
    else:
        MC = MonteCarlo_Control()
        num_ep = 500000
        for i in range(num_ep):
            MC.control()
        Q_star = MC.Q
        
        with open(filename, 'wb') as f:
            pickle.dump(Q_star,f)
    print('Q_star Loaded')

    Control = Sarsa_Control()
    wins = 0
    num_ep = 1000
    for i in np.linspace(0,1,11):
        Control.TD_learn()
    print(f'Winner Prob = {wins/num_ep * 100} %')
    #MC.playGame(n=10000)py
    MC.plot_Q()