import time
from nim_state_manager import StateManager
import random
import math
from copy import deepcopy

class MCTS:
    def __init__(self, sm):
        self.sm = sm
        self.root = Node()
        self.root.update_n()
        self.wins = {'player0': 0, 'player1': 0}
        self.num_rollouts = 0
        self.num_runs = 0
        self.prob_from_root = {}
        self.winner_states = 0

    def loop(self, time_limit, max_runs):
        timer = True
        start = time.time()
        while self.num_runs < max_runs:
            sm = deepcopy(self.sm)
            node, sm = self.select(sm)
            if (self.expand(node, sm)):
                node = random.choice(node.get_children())
                sm.move(node.get_action())
                if sm.get_winner() is not None:
                    winner = sm.get_winner()
                else:
                    winner = self.rollout(sm)
            else:
                winner = sm.get_winner()
            self.backprop(node, winner)

            current_time = time.time()
            if current_time - start > time_limit:
                timer = False

            self.num_runs += 1
        
        if self.root.has_children():
            for child in self.root.get_children():
                self.prob_from_root[child.get_action()] = child.get_num_visited()

        return max(self.prob_from_root, key = self.prob_from_root.get)

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
                temp = child.get_q() + (100 * self.uct(node, child))
                if temp > current:
                    best_child = child
                    current = temp
            node = best_child
            sm.move(node.get_action())
            nodes_selected += 1
        return node, sm

    def expand(self, node, sm):
        parent_node = node
        legal_actions = sm.get_possible_moves()
        if parent_node.player == 0:
            curr_player = 1
        else:
            curr_player = 0
        if len(legal_actions) > 0:
            for action in legal_actions:
                node.children.append(Node(action, parent_node, curr_player))
            return True
        else:
            return False

    # def rollout(self, sm):
    #     self.num_rollouts += 1
    #     sm_rollout = deepcopy(sm)
    #     total_score = 0
    #     for i in range(10):
    #         no_winner = True
    #         sm_rollout = deepcopy(sm)
    #         while no_winner:
    #             moves = sm_rollout.get_possible_moves()
    #             move = random.choice(moves)

    #             if sm_rollout.move(move):
    #                 winner = sm_rollout.winner
    #                 no_winner = False
    #                 if winner == 0:
    #                     score = 1
    #                 else:
    #                     score = -1
    #         total_score += score
    #     return total_score


    def rollout(self, sm):
        self.num_rollouts += 1
        sm = sm
        while True:
            moves = sm.get_possible_moves()
            move = random.choice(moves)

            if sm.move(move):
                return sm.get_winner()

    def backprop(self, node, winner):
        outcome = 1 if winner == 0 else -1

        while node.parent is not None:
            node.update_eval(outcome)
            node.update_n()
            node.update_q()
            node = node.parent

        node.update_eval(outcome)
        node.update_n()
        node.update_q()

class Node:
    def __init__(self, action=None, parent=None, player=0):
        self.parent = parent  # parent node
        self.action = action  # action to get here
        self.num_visited = 0  # number of times node has been visited
        self.q_value = 0  # the current value of node state
        self.eval = 0
        self.children = []  # children nodes
        self.player = player

    def has_children(self):
        if len(self.children) > 0:
            return True

    def update_eval(self, outcome):
        self.eval += outcome

    def update_n(self):
        self.num_visited += 1

    def update_q(self):
        self.q_value = self.eval / self.num_visited

    def get_state(self):
        return self.state

    def get_action(self):
        return self.action

    def get_children(self):
        return self.children

    def get_num_visited(self):
        return self.num_visited

    def get_eval(self):
        return self.eval

    def get_q(self):
        return self.q_value

    def get_parent(self):
        return self.parent



if __name__ == "__main__":
    sm = StateManager(10)
    mcts = MCTS(sm)
    suggested_move = mcts.loop(5, 500)
    print(f'win distribution: {mcts.wins}')
    print(f'runs: {mcts.num_runs}')
    print(f'rollouts: {mcts.num_rollouts}')
    print(f'root state: {mcts.sm.get_state()}')
    print(f'probabilities of actions from root: {mcts.prob_from_root}')
    print(f'suggested move: {suggested_move}')
    print(f'root visits: {mcts.root.get_num_visited()}')

    

    
