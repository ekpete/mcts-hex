from anet import TOPP_agent
from hex import StateManager
import random
import torch.nn as nn 

#play one game between two players
def play(player1, player2, size):
    game = StateManager(size)
    game.move(game.get_player(), random.choice(game.get_possible_moves()))
    while game.winner is None:
        game.move(game.get_player(), player2.get_move(game.get_flattened_board()))
        if game.winner is None:
            game.move(game.get_player(), player1.get_move(game.get_flattened_board()))
    return game.winner

#play one series of G games between two players
def series(player1, player2, num, size):
    wins={player1.name:0, player2.name:0}
    for i in range(1, num+1):
            winner = play(player1, player2, size)
            if winner == 1:
                wins[player1.name]+=1
            if winner == 2:
                wins[player2.name]+=1
    return wins

#play a tournament between all players
def tournament(players, G, board_size):
    for player in players:
        for i in range(len(players)):
            if i != players.index(player):
                print(str(series(player, players[i], int(G/2), board_size)).replace("{","").replace("}", "").replace("'","").replace(","," -"))

def decision(probability):
    return random.random() < probability

def play_saved_models(G):
    board_size = 5
    layers=[(25, nn.ReLU()),(25, None)]
    t0 = TOPP_agent(board_size, layers, "saved_models/actor_0.pt")
    t1 = TOPP_agent(board_size, layers, "saved_models/actor_100.pt")
    t2 = TOPP_agent(board_size, layers, "saved_models/actor_200.pt")
    t3 = TOPP_agent(board_size, layers, "saved_models/actor_300.pt")
    t4 = TOPP_agent(board_size, layers, "saved_models/actor_400.pt")
    players = [t0, t1, t2, t3, t4]
    tournament(players, G, board_size)

def play_saved_models_demo(G):
    board_size = 5
    layers = [(25, nn.ReLU()),(25, None)]
    t0 = TOPP_agent(board_size, layers, "saved_models_demo/actor_0.pt")
    t1 = TOPP_agent(board_size, layers,  "saved_models_demo/actor_12.pt")
    t2 = TOPP_agent(board_size, layers,"saved_models_demo/actor_24.pt")
    t3 = TOPP_agent(board_size, layers, "saved_models_demo/actor_36.pt")
    t4 = TOPP_agent(board_size, layers, "saved_models_demo/actor_48.pt")
    players = [t0, t1, t2, t3, t4]
    tournament(players, G, board_size)


if __name__ == "__main__":
    #play_saved_models(50)
    play_saved_models_demo(50)

