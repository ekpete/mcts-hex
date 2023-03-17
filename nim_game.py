from nim import StateManager
from mcts import MCTS
import random

class Player:
    def __init__(self, agent = False):
        self.agent = agent

    def move(self, state):
        if self.agent:
            mcts = MCTS(10)
            mcts.loop(board_size=state)
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
        game.move(game.get_player(), mcts_agent.move(game.get_state()))
        if game.get_state() > 0:
            game.move(game.get_player(), random_agent.move(game.get_state()))
        
    return game.winner

if __name__ == "__main__":
    wins = {'player1': 0, 'player2': 0}
    games = 50
    for i in range(games):
        wins[f'player{play()}'] += 1
        if i % 1 == 0:
            print(f'{i} games done. {games-i} left.')
    print(wins)

