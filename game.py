from hex import StateManager, update_board
from mcts import MCTS
from tkinter import Tk, Canvas
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
    return game.winner

if __name__ == "__main__":
    board_size = 4
    games = 1
    root = Tk()
    root.title("Hex")
    C = Canvas(root, bg="white", height=board_size*80, width=board_size*100)
    C.pack()
    wins = {'player1': 0, 'player2': 0}
    rbuf = {'board':[], 'action_probs':[]}
    for i in range(games):
        player = play(board_size, rbuf, root, C)
        wins[f'player{player}'] += 1
        print(f'{i} games done. {games-i} left.')
    print(wins)
    print(rbuf)
    root.mainloop()


