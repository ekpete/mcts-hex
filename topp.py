from anet import TOPP_agent
from hex import StateManager, gui_update_board, gui_print_board
from tkinter import Tk, Canvas
import time
import random

def play(player1, player2, size, root, C):

    game = StateManager(size)
    #gui_update_board(root, C, game)

    game.move(game.get_player(), random.choice(game.get_possible_moves()))
    #gui_update_board(root, C, game)
    #time.sleep(0.5)
    
    while game.winner is None:
        game.move(game.get_player(), player2.get_move(game.get_flattened_board()))
        #gui_update_board(root, C, game)
        #time.sleep(0.5)
        if game.winner is None:
            game.move(game.get_player(), player1.get_move(game.get_flattened_board()))
            #gui_update_board(root, C, game)
            #time.sleep(0.5)
    return game.winner

def series(player1, player2, num, size, root, C):
    wins={'player1':0, 'player2':0}
    for i in range(1, num+1):
            winner = play(player1, player2, size, root, C)
            if winner == 1:
                wins[f'player1']+=1
            if winner == 2:
                wins[f'player2']+=1
    return wins

if __name__ == "__main__":
    root = Tk()
    root.title('Hex')
    C = Canvas(root, bg='white', height=5*80, width=5*100)
    C.pack()
    m = 100
    t0 = TOPP_agent(5, "saved_models/model_game_0.pt")
    t1 = TOPP_agent(5, "saved_models/model_game_100.pt")
    t2 = TOPP_agent(5, "saved_models/model_game_200.pt")
    t3 = TOPP_agent(5, "saved_models/model_game_300.pt")
    t4 = TOPP_agent(5, "saved_models/model_game_400.pt")
    print("t4 v t0:")
    print(series(t4,t0,m,5, root, C))
    print('---------')
    print("t3 v t0:")
    print(series(t3,t0,m,5, root, C))
    print('---------')
    print("t2 v t0:")
    print(series(t2,t0,m,5, root, C))
    print('---------')
    print("t1 v t0:")
    print(series(t1,t0,m,5, root, C))
    print('---------')
    print("t4 v t1:")
    print(series(t4,t1,m,5, root, C))
    print('---------')
    print("t4 v t2:")
    print(series(t4,t2,m,5, root, C))
    print('---------')
    print("t4 v t3:")
    print(series(t4,t3,m,5, root, C))
    print('---------')
    print("t3 v t2:")
    print(series(t3,t2,m,5, root, C))
    print('---------')
    print("t3 v t1:")
    print(series(t3,t1,m,5, root, C))
    print('---------')
    print("t2 v t1:")
    print(series(t2,t1,m,5, root, C))
    print('---------')