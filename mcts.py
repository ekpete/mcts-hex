import time
from nim_state_manager import StateManager
import random
import math
from copy import deepcopy

class MCTS:
    def __init__(self, sm, c=1):
        self.sm = sm
        self.c = c # exploration factor
        self.root = Node()
        self.num_rollouts = 0
        self.rollout_wins = {'player0': 0, 'player1': 0} # win tally
        self.prob_from_root = {}

    def loop(self, time_limit, max_rollouts):
        timer = True
        start = time.time()
        while self.num_rollouts < max_rollouts:
            sm = deepcopy(self.sm)
            node, sm = self.select(sm)
            if (self.expand(node, sm)):
                node = random.choice(node.get_children())
                sm.move(node.get_action())
            
            winner = self.rollout(sm)
            self.backprop(node, winner)

            self.rollout_wins[f'player{winner}'] += 1
            current_time = time.time()
            if current_time - start > time_limit:
                timer = False
        
        if self.root.has_children():
            for child in self.root.get_children():
                self.prob_from_root[child.get_action()] = child.get_num_visited()


    def uct(self, node, child_node):
        return math.sqrt((math.log(node.get_num_visited()))/(1 + child_node.get_num_visited()))

    def select(self, sm):
        sm = sm
        node = self.root
        nodes_selected = 0
   
        while node.has_children():
            current = -math.inf
            for child in node.get_children():
                if child.get_num_visited() == 0:
                    sm.move(child.get_action())
                    return child, sm
                temp = child.get_q() + (self.c * self.uct(node, child))
                if temp > current:
                    best_child = child
                    current = temp
            node = best_child
            sm.move(node.get_action())
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
        sm = sm

        if sm.winner is not None:
            return sm.get_winner()
        
        while True:
            moves = sm.get_possible_moves()
            move = random.choice(moves)
            if sm.move(move):
                return sm.get_winner()

    def backprop(self, node, winner):
        outcome = 1 if winner == 0 else -1
        while node.get_parent() is not None:
            node.update(outcome)
            node = node.get_parent()
        node.update(outcome)

    def get_action_probs(self):
        return self.prob_from_root

    def get_best_move(self):
        if self.prob_from_root:
            return max(self.prob_from_root, key = self.prob_from_root.get)

class Node:
    def __init__(self, action=None, parent=None):
        self.parent = parent  # parent node
        self.action = action  # action to get here
        self.num_visited = 0  # number of times node has been visited
        self.q_value = 0  # the current value of node state
        self.eval = 0
        self.children = []  # children nodes

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
    sm = StateManager(10)
    mcts = MCTS(sm, 10)
    suggested_move = mcts.loop(5, 1000)
    print(f'win distribution: {mcts.rollout_wins}')
    print(f'rollouts: {mcts.num_rollouts}')
    print(f'root state: {mcts.sm.get_state()}')
    print(f'probabilities of actions from root: {mcts.get_action_probs()}')
    print(f'suggested move: {mcts.get_best_move()}')

    

    
