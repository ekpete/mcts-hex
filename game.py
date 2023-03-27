from hex import StateManager, gui_update_board, gui_print_board
from mcts import MCTS
from anet import ANET
from tkinter import Tk, Canvas
import torch.nn as nn 
import torch.optim as optim 
from tqdm import tqdm
import math

def play(size, max_rollouts, rbuf, root, C, c):
    mcts = MCTS(c)
    game = StateManager(size)
    gui_update_board(root, C, game)
    while game.get_winner() is None:
        mcts.loop(board_size=game.board.board_size,max_rollouts=max_rollouts, board=game.get_board_state(), player=game.get_player())
        rbuf['board'].append(game.get_flattened_board())
        rbuf['action_probs'].append(mcts.get_visit_probs())
        action = mcts.get_best_move()
        game.move(game.get_player(), action)
        mcts.prune_tree(action)
        gui_update_board(root, C, game)
    #print(f'Rollout games this game: {mcts.total_rollouts}')
    return game.winner

def games(settings):
    games = settings['Number of RL episodes']
    board_size = settings['Board size']
    max_rollouts = settings['Max rollout games']
    c = settings['Exploration factor']
    anet_input = (board_size**2)+1
    learning_rate = settings['ANET learning rate']
    layers_data = settings['ANET layers data']
    optimizer = settings['ANET optimizer']
    batch_size = settings['ANET batch size']
    save = settings['Save ANETs']
    m = settings['M cached ANETs']-1

    root = Tk()
    root.title('Hex')
    C = Canvas(root, bg='white', height=board_size*80, width=board_size*100)
    C.pack()

    rbuf = {'board':[], 'action_probs':[]}
    wins = {'player1': 0, 'player2': 0}

    save_interval = games/m
    anet = ANET(input_size=anet_input, layers=layers_data, learning_rate=learning_rate, optimizer=optimizer)
    anet.save_model(0)

    board1 = (1,1,1,2,0,0,2,1,2,1,0,2,1,1,2,0,2,0,0,0,0,0,0,0,0,0)
    legal = (0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1)

    for g in tqdm(range(games), desc="Progress:"):
        winner = play(board_size, max_rollouts, rbuf, root, C, c)
        if len(rbuf['board'])>batch_size:
            anet.train(rbuf, batch_size)
        if (int(g+1) % int(save_interval) == 0) and save:
            anet.save_model(g+1)
            #print(anet.get_best_move(board1, legal))
        wins[f'player{winner}'] += 1

    print(f"\nPlayer 1: {wins['player1']} wins.\nPlayer 2: {wins['player2']} wins.")
    
    anet.print_losses()
    bord1_gui = [[1,1,2,0,0],[2,1,2,1,0],[2,1,1,2,0],[2,0,0,0,0],[0,0,0,0,0]]
  

    C.delete('all')
    gui_print_board(C, bord1_gui,[])
    root.update_idletasks()
    root.update()

    print(f"RBUF size: {len(rbuf['board'])}")
    print(anet.predicted_action_probs(board1, legal))
   
    
    root.mainloop()


if __name__ == "__main__":
    settings = {
        'Number of RL episodes': 400,
        'Board size': 5,
        'Max rollout games': 500,
        'Exploration factor': 1,
        'ANET learning rate': 0.01,
        'ANET layers data': [(25, nn.ReLU()),(25, None)],
        'ANET optimizer': optim.Adam,
        'ANET batch size': 128,
        'Save ANETs': True,
        'M cached ANETs': 5,
    }
    games(settings)
            
