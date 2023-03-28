from anet import TOPP_agent
from hex import StateManager, gui_update_board, gui_print_board
from tkinter import Tk, Canvas
import time
import random

def play(player1, player2, size):
    game = StateManager(size)
    game.move(game.get_player(), random.choice(game.get_possible_moves()))
    while game.winner is None:
        game.move(game.get_player(), player2.get_move(game.get_flattened_board()))
        if game.winner is None:
            game.move(game.get_player(), player1.get_move(game.get_flattened_board()))
    return game.winner

def series(player1, player2, num, size):
    wins={player1.name:0, player2.name:0}
    for i in range(1, num+1):
            winner = play(player1, player2, size)
            if winner == 1:
                wins[player1.name]+=1
            if winner == 2:
                wins[player2.name]+=1
    return wins

def tournament(players, G):
    for player in players:
        for i in range(len(players)):
            if i != players.index(player):
                print(series(player, players[i], int(G/2), 5))

if __name__ == "__main__":
    G = 50
    t0 = TOPP_agent(5, "saved_models/actor_0.pt")
    t1 = TOPP_agent(5, "saved_models/actor_100.pt")
    t2 = TOPP_agent(5, "saved_models/actor_200.pt")
    t3 = TOPP_agent(5, "saved_models/actor_300.pt")
    t4 = TOPP_agent(5, "saved_models/actor_400.pt")
    players = [t0, t1, t2, t3, t4]
    tournament(players, G)