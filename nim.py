from mcts import *
import random

class StateManager:
    def __init__(self, start_state, board=None, player = 1):
        self.state = start_state
        self.player = player
        self.k = 3
        self.winner = None

    def switch_player(self):
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1
    
    def get_player(self):
        return self.player
    
    def get_winner(self):
        return self.winner
    
    def move(self, player, count):
        if self.winner is not None:
            return True
        self.state -= count
        if self.state == 0:
            self.winner = self.player
            return True
        self.switch_player()
        return False
    
    def get_state(self):
        return self.state

    def get_possible_moves(self):
        if self.state <= 0:
            return []
        elif self.state == 1:
            return [1]
        elif self.state == 2:
            return [1,2]
        elif self.state == 3:
            return [1,2,3]
        else:
            moves = []
            for i in range(self.k):
                moves.append(i + 1)
            return moves

class Player:
    def __init__(self, agent = False):
        self.agent = agent

    def move(self, state):
        if self.agent:
            mcts = MCTS()
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
    game = StateManager(13)
    mcts_agent = Player(True)
    random_agent = Player(True)
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

