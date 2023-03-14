from tkinter import *
import math

class StateManager:
    def __init__(self, board_size = 7, positions = None):
        self.board = Board(board_size, positions)

class Board:
    def __init__(self, board_size = 7, positions = None):
        self.board_size = board_size
        self.positions = positions
        

class Piece:
    def __init__(self, player = 0, board_location = None):
        self.player = player
        self.board_location = board_location


def print_board(board):
    colours = ['white', 'dodgerblue', 'red']
    root = Tk()
    root.title("Hex")
    C = Canvas (root, bg="white", height=len(board)*80, width=len(board)*100)
    for i in range(len(board)):
        C.create_line(line('top', 15, 100+((2*i))*math.sqrt(3)*15, 100+(0*2)*((3/2)*15)), width = 4, fill = 'red')
        C.create_line(line('bottom', 15, 100+((len(board)-1)+(2*i))*math.sqrt(3)*15, 100+((len(board)-1)*2)*((3/2)*15)), width = 4, fill = 'red')
        C.create_line(line('left', 15, 100+((i))*math.sqrt(3)*15, 100+(i*2)*((3/2)*15)), width = 4, fill = 'dodgerblue')
        C.create_line(line('right', 15, 100+(((len(board)-1)*2)+(i))*math.sqrt(3)*15, 100+(i*2)*((3/2)*15)), width = 4, fill = 'dodgerblue')
        for j in range(len(board[0])):
            C.create_polygon(hexagonal(15,100+((j*2)+(i))*math.sqrt(3)*15,100+(i*2)*((3/2)*15)), fill = colours[board[i][j]], outline='black', width = 4)

    C.pack()
    root.mainloop()

def line(pos,cr,x,y):
    if pos == 'top':
        return -(math.sqrt(3)*cr)+(x), -cr+(y-10), 0+(x), -2*cr+(y-10), (math.sqrt(3)*cr)+(x), -cr+(y-10)
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
    board4 = [[1,0,2,0],[0,1,2,1],[0,0,2,0], [0,0,2,0]]
    print_board(board5)




