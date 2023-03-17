import random
from hex import StateManager, update_board
from tkinter import Tk, Canvas
import time

class Player:
    def __init__(self, player, agent = False):
        self.agent = agent
        self.player = player

    def move(self, state_manager):
        moves = state_manager.get_possible_moves()
        return random.choice(moves)
    
    def get_player(self):
        return self.player
    
def play(size):
    game = StateManager(size)
    player1 = Player(1)
    player2 = Player(2)

    root = Tk()
    root.title("Hex")
    C = Canvas(root, bg="white", height=size*80, width=size*100)
    C.pack()

    while game.get_winner() is None:
        game.move(player1.get_player() ,player1.move(game))
        update_board(root, C, game)
        time.sleep(0.1)

        if game.get_winner() is None:
            game.move(player2.get_player() ,player2.move(game))
            update_board(root, C, game)
            time.sleep(0.1)

    update_board(root, C, game)

    root.mainloop()

        
    return game.winner

if __name__ == "__main__":
    play(4)
