import random
from hex import StateManager, update_board
from mcts import MCTS
from tkinter import Tk, Canvas
import time

class Player:
    def __init__(self, player, agent = False):
        self.agent = agent
        self.player = player

    def move(self, state_manager):
        if self.agent:
            mcts = MCTS(1)
            mcts.loop(board_size=state_manager.board.board_size, board=state_manager.get_board_state(), player=state_manager.get_player())
            return mcts.get_best_move()
        else:
            moves = state_manager.get_possible_moves()
            return random.choice(moves)
    
    def get_player(self):
        return self.player
    
def play(size, root, C):
    game = StateManager(size)
    player1 = Player(1, True)
    player2 = Player(2, False)
    player1_moves = 0
    player2_moves = 0


    while game.get_winner() is None:
        game.move(player1.get_player() ,player1.move(game))
        player1_moves+=1
        update_board(root, C, game)
        time.sleep(0.1)

        if game.get_winner() is None:
            game.move(player2.get_player() ,player2.move(game))
            player2_moves+=1
            update_board(root, C, game)
            time.sleep(0.1)

    update_board(root, C, game)

        
    return game.winner, player1_moves, player2_moves

if __name__ == "__main__":
    board_size = 10

    root = Tk()
    root.title("Hex")
    C = Canvas(root, bg="white", height=board_size*80, width=board_size*100)
    C.pack()

    wins = {'player1': 0, 'player2': 0}
    player1_moves = 0
    player2_moves = 0
    games = 3
    for i in range(games):
        player, player1_m, player2_m = play(board_size, root, C)
        wins[f'player{player}'] += 1
        player1_moves += player1_m
        player2_moves += player2_m
        if i % 10 == 0:
            print(f'{i} games done. {games-i} left.')
    print(wins)
    print(f'Average moves: {player1_moves/games}')
