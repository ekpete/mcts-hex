from tkinter import *
import math
import time
from copy import deepcopy

class StateManager:
    def __init__(self, board_size = 7, positions = None):
        self.board = Board(board_size, positions)

class Board:
    def __init__(self, board_size = 7, positions = None):
        self.board_size = board_size
        if positions is None:
            self.board = self.make_empty_board(board_size)
        else:
            self.board = positions
    
    def make_empty_board(self, size):
        a = [[0 for _ in range(size)] for _ in range(size)]
        return a
    
    def place_piece(self, player, position):
        if self.board[position[0]][position[1]] == 0:
            self.board[position[0]][position[1]] = Piece(player, position)
            self.board[position[0]][position[1]].add_neighbours(self.board)
        else:
            print("occupied space")
    
    def simple_board(self):
        b = deepcopy(self.board)
        for i in range(len(b)):
            for j in range(len(b)):
                piece = b[i][j]
                b[i][j] = piece.player if isinstance(piece, Piece) else 0
        return b

    def legal_moves(self):
        moves = []
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[i][j] == 0:
                    moves.append((i,j))
        return moves
        
    def check_win(self):
        for piece in self.board[0]:
            if isinstance(piece, Piece) and piece.player == 2:
                chain = self.make_chain(piece)
                for piece in chain:
                    if piece.position[0] == (len(self.board)-1):
                        return 'winner is player 2', chain
        for i in range(len(self.board)):
            piece = self.board[i][0]
            if isinstance(piece, Piece) and piece.player == 1:
                chain = self.make_chain(piece)
                for piece in chain:
                    if piece.position[1] == (len(self.board)-1):
                        return 'winner is player 1', chain
        return 'no winner', []
    
    def make_chain(self, start_piece):
        chain = []
        chain.append(start_piece)
        for piece in chain:
            for n in piece.neighbours:
                if n not in chain:
                    chain.append(n)
        return chain


class Piece:
    def __init__(self, player = None, position = None):
        self.player = player
        self.position = position
        self.neighbours = []

    def add_neighbours(self, board):
        positions = (-1,0), (-1,1), (0,1), (1,0), (1, -1), (0, -1)
        for pos in positions:
            new_pos = (self.position[0] + pos[0], self.position[1] + pos[1])
            if new_pos[0] < 0 or new_pos[0] > len(board)-1 or new_pos[1] < 0 or new_pos[1] > len(board)-1:
                pass
            else:
                piece = board[new_pos[0]][new_pos[1]]
                if isinstance(piece, Piece) and piece.player == self.player:
                    self.neighbours.append(piece)
                    piece.neighbours.append(self)


def game_window(game, chain):
    root = Tk()
    root.title("Hex")
    C = Canvas(root, bg="white", height=len(game[0])*80, width=len(game[0])*100)
    C.pack()

    for board in game:
        C.delete('all')
        print_board(C, board, chain)
        root.update_idletasks()
        root.update()
        time.sleep(1)

    root.mainloop()

def print_board(C, board, chain):
    colours = ['white', 'dodgerblue', 'red']
    for i in range(len(board)):
        C.create_line(line('top', 15, 100+((2*i))*math.sqrt(3)*15, 100+(0*2)*((3/2)*15)), width = 4, fill = 'red')
        C.create_line(line('bottom', 15, 100+((len(board)-1)+(2*i))*math.sqrt(3)*15, 100+((len(board)-1)*2)*((3/2)*15)), width = 4, fill = 'red')
        C.create_line(line('left', 15, 100+((i))*math.sqrt(3)*15, 100+(i*2)*((3/2)*15)), width = 4, fill = 'dodgerblue')
        C.create_line(line('right', 15, 100+(((len(board)-1)*2)+(i))*math.sqrt(3)*15, 100+(i*2)*((3/2)*15)), width = 4, fill = 'dodgerblue')
        for j in range(len(board[0])):
            w = 8 if (i,j) in chain else 4
            C.create_polygon(hexagonal(15,100+((j*2)+(i))*math.sqrt(3)*15,100+(i*2)*((3/2)*15)), fill = colours[board[i][j]], outline='black', width = w)

def line(pos,cr,x,y):
    if pos == 'top':
        return -(math.sqrt(3)*cr)+(x), -cr+y-10, 0+x, -2*cr+(y-10), (math.sqrt(3)*cr)+x, -cr+y-10
    elif pos == 'bottom':
        return -(math.sqrt(3)*cr)+x, cr+y+10, 0+x, 2*cr+y+10, (math.sqrt(3)*cr)+x, cr+y+10
    elif pos == 'left':
        return -(math.sqrt(3)*cr)+x-7, -cr+y+7, -(math.sqrt(3)*cr)+x-7, cr+y+7, 0+x-7, 2*cr+y+7
    elif pos == 'right':
        return 0+x+7, -2*cr+y-7, (math.sqrt(3)*cr)+x+7, -cr+y-7, (math.sqrt(3)*cr)+x+7 , cr+y-7
    else:
        return 0,0,0

def hexagonal(cr, x, y):
    a = 0+x, -2*cr+y
    b = (math.sqrt(3)*cr)+x, -cr+y
    c = (math.sqrt(3)*cr)+x, cr+y
    d = 0+x, 2*cr+y
    e = -(math.sqrt(3)*cr)+x, cr+y
    f = -(math.sqrt(3)*cr)+x, -cr+y
    return a,b,c,d,e,f

if __name__ == "__main__":
    board8 = [[1,0,2,0,0,0,1,2],[0,1,2,1,1,2,0,0],[0,0,2,0,2,1,1,0], [1,2,1,0,0,2,0,1], [2,1,0,1,0,2,2,1], [2,0,0,1,1,2,1,2], [2,2,0,0,1,1,0,2],[0,2,2,0,2,1,1,0]]
    board5 = [[1,0,2,0,0],[0,1,2,1,1],[0,0,2,0,2], [0,0,2,0,1], [0,1,1,0,2]]
    board4 = [[1,1,2,0],[0,1,2,1],[0,0,2,0], [0,0,2,0]]
    board5_0 = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
    board5_1 = [[1,0,0,0,0],[0,0,0,0,0],[0,0,0,0,2], [0,0,0,0,0], [0,0,0,0,0]]
    board5_2 = [[1,0,0,0,0],[0,0,1,2,0],[0,0,0,0,2], [0,0,0,0,0], [0,0,0,0,0]]
    board5_3 = [[1,0,1,0,0],[0,0,1,2,0],[0,0,0,2,2], [0,0,0,0,0], [0,0,0,0,0]]
    board5_4 = [[1,0,1,0,0],[0,0,1,2,0],[0,0,1,2,2], [0,0,0,0,0], [0,0,2,0,0]]
    board5_5 = [[1,0,1,0,2],[0,0,1,2,0],[0,0,1,2,2], [0,0,1,0,0], [0,0,2,0,0]]
    board5_6 = [[1,0,1,0,2],[0,0,1,2,0],[0,0,1,2,2], [0,0,1,2,1], [0,0,2,0,0]]

    board5_7 = [[1,0,1,0,2],
                [0,0,1,2,0],
                [0,0,1,2,2], 
                [0,0,1,2,1], 
                [0,0,2,0,0]]

    game = [board5_0, board5_1, board5_2, board5_3, board5_4, board5_5, board5_6, board5_7]
    #game_window(game, [])

    board = Board(7)
    print(board.simple_board())
    board.place_piece(1, (0,0))
    print(board.legal_moves())
    board.place_piece(1, (0,1))
    board.place_piece(1, (1,3))
    print(board.check_win())
    board.place_piece(2, (2,3))
    board.place_piece(2, (1,3))
    board.place_piece(2, (1,1))
    board.place_piece(1, (0,2))
    board.place_piece(1, (0,3))
    board.place_piece(2, (2,2))
    board.place_piece(2, (3,3))
    board.place_piece(2, (2,1))
    board.place_piece(2, (3,1))
    board.place_piece(1, (1,0))
    board.place_piece(2, (4,0))
    board.place_piece(2, (5,0))
    board.place_piece(1, (1,4))
    board.place_piece(1, (0,5))
    board.place_piece(1, (0,6))
    board.place_piece(1, (6,0))
    board.place_piece(1, (6,1))
    #board.place_piece(2, (6,0))
    print('-------------')
    print(board.legal_moves())
    print(board.simple_board())
    winner, chain = board.check_win()
    print(winner)
    new_chain = [x.position for x in chain]
    game_window([board.simple_board()], new_chain)

