from nim_state_manager import StateManager
from mcts import MCTS
import random

class Player:
    def __init__(self, agent = False):
        self.agent = agent

    def move(self, state):
        if self.agent:
            sm = StateManager(state)
            mcts = MCTS(sm,10)
            mcts.loop(5,1000)
            return mcts.get_best_move()
        else:
            if state == 1:
                return 1
            elif state == 2:
                return 2
            elif state == 3:
                return 3
            else:
                return random.randint(1,3)

def play():
    game = StateManager(20)
    mcts_agent = Player(True)
    random_agent = Player(False)
    while game.winner is None:
        game.move(mcts_agent.move(game.get_state()))
        if game.get_state() > 0:
            game.move(random_agent.move(game.get_state()))
        
    return game.winner
   

if __name__ == "__main__":
    wins = {'player0': 0, 'player1': 0}
    games = 10
    for i in range(games):
        wins[f'player{play()}'] += 1
        if i % 1 == 0:
            print(f'{i} games done. {games-i} left.')
    print(wins)

