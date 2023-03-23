from hex import StateManager, update_board
from mcts import MCTS
from anet import ANET
from tkinter import Tk, Canvas
import torch.nn as nn 
import torch.optim as optim
from copy import deepcopy    

def play(size, rbuf, root, C):
    mcts = MCTS()
    game = StateManager(size)
    update_board(root, C, game)
    while game.get_winner() is None:
        mcts.loop(board_size=game.board.board_size, board=game.get_board_state(), player=game.get_player())
        rbuf['board'].append(game.get_flattened_board())
        rbuf['action_probs'].append(mcts.get_visit_probs())
        action = mcts.get_best_move()
        game.move(game.get_player(), action)
        mcts.prune_tree(action)
        update_board(root, C, game)
    print(f'Total rollout games: {mcts.total_rollouts}')
    return game.winner

def games(settings):
    board_size = settings["board_size"]
    games = settings["num_episodes"]
    root = Tk()
    root.title("Hex")
    C = Canvas(root, bg="white", height=board_size*80, width=board_size*100)
    C.pack()
    g = 0
    rbuf = {'board':[], 'action_probs':[]}
    wins = {'player1': 0, 'player2': 0}
    save_interval = games/settings["M_cached_anets"]
    anet = ANET(input_size=board_size*board_size+1, layers=settings["layers_data"], learning_rate=settings["learning_rate"])
    anet.save_model(g)
    for i in range(games):
        g+=1
        player = play(board_size, rbuf, root, C)
        wins[f'player{player}'] += 1
        print(f'{i} games done. {games-i} left.')
        if len(rbuf['board'])>64:
            anet.train(rbuf, 64)
        if g % save_interval == 0:
            pass
            #anet.save_model(g)
    
    board1 = (2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 0, 0, 0, 0)
    legal = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1)
    print(anet.get_best_move(board1, legal))
    anet.print_losses()
    
    print(wins)
    print(f"{len(rbuf['board'])}")
    root.mainloop()


if __name__ == "__main__":
    settings = {
        "num_episodes": 50,
        "board_size": 5,
        "rollout_games": 500,
        "learning_rate": 0.01,
        "layers_data": [(30, nn.ReLU()),(30, nn.ReLU()), (25, None)],
        "optimizer": optim.Adam,
        "batch_size": 64,
        "M_cached_anets": 5,
    }
    games(settings)

    moves = []
    k=5

    for i in range(k):
        for j in range(k):
            moves.append((i,j))
    print(moves[24])
            



