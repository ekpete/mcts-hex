import math
from copy import deepcopy

class StateManager:
    def __init__(self, board_size = 4, positions = None, player = 1):
        self.board = Board(board_size, positions, player)
        self.player = player # current players turn
        self.winner = None
        self.win_chain = []
    
    def get_player(self):
        return self.player
    
    def get_winner(self):
        return self.winner
    
    def get_win_chain(self):
        new_chain = [x.position for x in self.win_chain]
        return new_chain
    
    def move(self, player, position):
        if self.winner is not None:
            return True
        self.board.place_piece(player, position)
        win, player, chain = self.board.check_win()
        if win:
            self.winner = player
            self.win_chain = chain
            return True

        self.player = self.board.get_current_player()
        return False

    def get_state(self):
        return self.board.get_simple_board()
    
    def get_board_state(self):
        return self.board.get_board()

    def get_possible_moves(self):
        return self.board.legal_moves()

class Board:
    def __init__(self, board_size = 4, positions = None, player = 1):
        self.board_size = board_size
        if positions is None:
            self.board = self.make_empty_board(board_size)
        else:
            self.board = positions
        self.player = player
    
    def make_empty_board(self, size):
        a = [[0 for _ in range(size)] for _ in range(size)]
        return a
    
    def place_piece(self, player, position):
        if player == self.player:
            if self.board[position[0]][position[1]] == 0:
                self.board[position[0]][position[1]] = Piece(player, position)
                self.board[position[0]][position[1]].add_neighbours(self.board)
                self.player = 2 if self.player == 1 else 1
                print(f'move placed, next player is player {self.player}')
            else:
                print("occupied space")
        else:
            print(f"Not player {player}'s turn. Next to move is player {self.player}")
        
    def legal_moves(self):
        moves = []
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[i][j] == 0:
                    moves.append((i,j))
        return moves
        
    def get_board(self):
        return self.board
    
    def get_simple_board(self):
        b = deepcopy(self.board)
        for i in range(len(b)):
            for j in range(len(b)):
                piece = b[i][j]
                b[i][j] = piece.player if isinstance(piece, Piece) else 0
        return b
    
    def get_board_size(self):
        return self.board_size
    
    def get_current_player(self):
        return self.player
        
    def check_win(self):
        for piece in self.board[0]:
            if isinstance(piece, Piece) and piece.player == 2:
                chain = self.make_chain(piece)
                for piece in chain:
                    if piece.position[0] == (len(self.board)-1):
                        return True, 2, chain
        for i in range(len(self.board)):
            piece = self.board[i][0]
            if isinstance(piece, Piece) and piece.player == 1:
                chain = self.make_chain(piece)
                for piece in chain:
                    if piece.position[1] == (len(self.board)-1):
                        return True, 1, chain
        return False, None, []
    
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

"""Functions to print Hex board"""

def update_board(root, C, game):
    C.delete('all')
    print_board(C, game.get_state(), game.get_win_chain())
    w = game.get_winner()
    if w == 1 or w == 2:
        C.create_text(C.winfo_width()/2,C.winfo_height()-20,fill="black",font="Helvetica 15 bold", text=f"Player {w} wins!")
    root.update_idletasks()
    root.update()

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
        return -(math.sqrt(3)*cr)+(x), -cr+y-10, 0+x, -2*cr+y-10, (math.sqrt(3)*cr)+x, -cr+y-10
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
