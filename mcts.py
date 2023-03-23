import time
from hex import StateManager
import random
import math
from copy import deepcopy

class MCTS:
    def __init__(self, c=1):
        self.sm = None # state manager
        self.c = c # exploration factor
        self.root = Node()
        self.num_rollouts = 0
        self.total_rollouts = 0
        self.rollout_wins = {'player1': 0, 'player2': 0}
        self.visits_from_root = {}
        self.Qs_from_root = {}

    def loop(self, board_size, time_limit=10, max_rollouts=1000, board=None, player=1):
        timer = True
        start = time.time()
        self.board_size = board_size
        self.sm = StateManager(board_size, board, player)
        while self.num_rollouts < max_rollouts:
            sm = deepcopy(self.sm)
            node, sm = self.select(sm)
            if (self.expand(node, sm)):
                node = random.choice(node.get_children())
                sm.move(sm.get_player(), node.get_action())
            winner = self.rollout(sm)
            self.backprop(node, winner)
            self.rollout_wins[f'player{winner}'] += 1
            current_time = time.time()
            if current_time - start > time_limit:
                timer = False
        if self.root.has_children():
            for child in self.root.get_children():
                self.visits_from_root[child.get_action()] = child.get_num_visited()
                self.Qs_from_root[child.get_action()] = child.get_q()
        self.num_rollouts = 0

    def select(self, sm):
        sm = sm
        node = self.root
        nodes_selected = 0
        while node.has_children():
            current1 = -math.inf
            current2 = math.inf
            for child in node.get_children():
                if child.get_num_visited() == 0:
                    sm.move(sm.get_player(), child.get_action())
                    return child, sm
                if sm.get_player() == 1:
                    temp = child.get_q() + (self.c * self.uct(node, child))
                    if temp > current1:
                        best_child = child
                        current1 = temp
                elif sm.get_player() == 2:
                    temp = child.get_q() - (self.c * self.uct(node, child))
                    if temp < current2:
                        best_child = child
                        current2 = temp
            node = best_child
            sm.move(sm.get_player(), node.get_action())
            nodes_selected += 1
        return node, sm

    def expand(self, node, sm):
        legal_actions = sm.get_possible_moves()
        if len(legal_actions) > 0:
            for action in legal_actions:
                node.children.append(Node(action, node))
            return True
        else:
            return False

    def rollout(self, sm):
        self.num_rollouts += 1
        self.total_rollouts += 1
        sm = sm
        if sm.winner is not None:
            return sm.get_winner()
        while True:
            moves = sm.get_possible_moves()
            move = random.choice(moves)
            if sm.move(sm.get_player(), move):
                return sm.get_winner()

    def backprop(self, node, winner):
        outcome = 1 if winner == 1 else -1
        while node.get_parent() is not None:
            node.update(outcome)
            node = node.get_parent()
        node.update(outcome)
    
    def prune_tree(self, action):
        for child in self.root.get_children():
            if child.get_action() == action:
                self.root = child
                self.root.parent = None
        self.visits_from_root = {}
        self.Qs_from_root = {}
    
    def uct(self, node, child_node):
        return math.sqrt((math.log(node.get_num_visited()))/(1 + child_node.get_num_visited()))

    def get_action_visits(self):
        return self.visits_from_root
    
    def get_action_Qs(self):
        return self.Qs_from_root

    def get_best_move(self):
        if self.visits_from_root:
            return max(self.visits_from_root, key = self.visits_from_root.get)
    
    def get_visit_probs(self):
        visits = deepcopy(self.visits_from_root)
        summ = sum(visits.values())
        k = self.board_size
        for i in range(k):
            for j in range(k):
                if (i,j) in visits:
                    visits[(i,j)] = visits[(i,j)]/summ
                else:
                    visits[(i,j)] = 0
        return tuple(dict(sorted(visits.items())).values())

class Node:
    def __init__(self, action=None, parent=None):
        self.parent = parent  
        self.action = action  # action to get here
        self.num_visited = 0  # N(s,a) - visit count
        self.q_value = 0  # Q(s,a) - q value
        self.eval = 0 # E - rewards
        self.children = [] 
    def has_children(self):
        if len(self.children) > 0:
            return True

    def update(self, outcome):
        self.eval += outcome
        self.num_visited += 1
        self.q_value = self.eval / self.num_visited

    def get_action(self):
        return self.action

    def get_children(self):
        return self.children

    def get_num_visited(self):
        return self.num_visited

    def get_q(self):
        return self.q_value

    def get_parent(self):
        return self.parent


if __name__ == "__main__":
    game = StateManager(3)
    mcts = MCTS()
    mcts.loop(board_size=game.board.board_size, board=game.get_board_state(), player=game.get_player())
    print(f'win distribution: {mcts.rollout_wins}')
    print(f'rollouts: {mcts.num_rollouts}')
    print(f'root state: {mcts.sm.get_state()}')
    print(f'visits of actions from root: {mcts.get_action_visits()}')
    print(f'Qs of actions from root: {mcts.get_action_Qs()}')
    print(f'suggested move: {mcts.get_best_move()}')
    suggested_move = mcts.get_best_move()
    mcts.prune_tree(suggested_move)
    game.move(1, suggested_move)
    print(game.get_board_state())
    print(game.get_player())
    mcts.loop(board_size=game.board.board_size, board=game.get_board_state(), player=game.get_player())
    print(f'visits of actions from root: {mcts.get_action_visits()}')
    print(f'suggested move: {mcts.get_best_move()}')

    
