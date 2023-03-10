class StateManager:
    def __init__(self, start_state):
        self.state = start_state
        self.player = 0
        self.k = 3
        self.winner = None

    def switch_player(self):
        if self.player == 0:
            self.player = 1
        else:
            self.player = 0
    
    def get_player(self):
        return self.player
    
    def get_winner(self):
        return self.winner
    
    def move(self, count):
        if self.winner is not None:
            return True

        self.state -= count
        if self.state == 0:
            self.winner = self.player
            return True
        self.switch_player()
        return False
    
    def get_state(self):
        return self.state

    def get_possible_moves(self):
        if self.state <= 0:
            return []
        elif self.state == 1:
            return [1]
        elif self.state == 2:
            return [1,2]
        elif self.state == 3:
            return [1,2,3]
        else:
            moves = []
            for i in range(self.k):
                moves.append(i + 1)
            return moves

if __name__ == "__main__":
    game = StateManager(20)
    print(game.get_state())
    print(game.get_possible_moves())
    print(game.get_player())
    print(game.move(2))
    print(game.get_player())
    print(game.get_state())
    game.move(3)
    print(game.get_state())
    game.move(3)
    print(game.get_state())
    game.move(3)
    print(game.get_state())
    game.move(3)
    print(game.get_state())
    game.move(3)
    print(game.get_state())
    game.move(2)
    print(game.get_state())
    print(game.winner)
    game.move(3)
    print(game.get_state())
    print(game.winner)

