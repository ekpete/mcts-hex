class Nim:
    def __init__(self, k=3, n=21):
        self.k = k
        self.n = n
        self.player = 0 
        self.winner = None
    
    def switch_player(self):
        if self.player == 0:
            self.player = 1
        else:
            self.player = 0
    
    def move(self, count):
        if self.winner is not None:
            raise Exception("Game finished")
        if count > self.k or count <1:
            raise Exception("Too many removed")

        self.n -= count
        if self.n == 0:
            self.winner = self.player
        self.switch_player()

        
def play():
    game = Nim()
    while True:
        print(f"Stones left: {game.n}")
        print(f"Player {game.player}'s turn.")
        count = int(input("Remove 1,2 or 3 stones: "))
        game.move(count)
        if game.winner is not None:
            print(f"Game over. Winner is player {game.winner}!")
            return



def main():
    play()

if __name__ == "__main__":
    main()