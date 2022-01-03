import numpy as np
import random
#Can optimise by setting black to 1 and red to -1

class Easy21():
    def __init__(self):
        self.player_value = random.randint(1,10)
        self.dealer_value = random.randint(1,10) 
        self.state = self.get_state(), None

    def get_state(self):
        return (self.player_value,self.dealer_value)
    def step(self,state,a):
        if a == 0: #hit : 0, stick : 1
            self.player_value += self.draw_card()
            if self.isBust(self.player_value):
                return self.get_state() , -1
            else:
                return self.get_state(), None

        else: # If Stick do dealers turn
            while(not self.isBust(self.dealer_value)):
                if self.dealer_value < 17:
                    self.dealer_value += self.draw_card()
                else:
                    # Terminal State, also calculate the winner 
                    return self.get_state(),np.sign(self.player_value - self.dealer_value)
            return self.get_state(), 1
    def draw_card(self):
        return random.choices(range(-10,11), weights= [1]*10 +[0] + [2]*10, k =1)[0]
    def isBust(self,value):
        return (value < 1 or value > 21)
    def new_Game(self):
        self.player_value = random.randint(1,10)
        self.dealer_value = random.randint(1,10) 
        self.state = self.get_state(), None

    
#Run Tests

if __name__ == "__main__":
    def main():
        random.seed(a=42)
        game= Easy21()
        print(game.state)
        for i in range(100):
            a = input('Action ( hit/stick ): ')
            game.state, r = game.step(state = game.state,a = a)
            print(game.state, r)
            if r is not None:
                print('new game')
                game.__init__()
                print(game.state)
    #main()
    Easy21().hello()