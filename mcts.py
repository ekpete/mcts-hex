import time
from gamestate import StateManager
import random
import math
from copy import deepcopy

class Node:
    def __init__(self, action=None, parent=None):
        self.parent = parent # parent node
        self.action = action # action to get here
        self.num_visited = 0 # number of times node has been visited
        self.q_value = 0 # the current value of node state
        self.eval = 0
        self.children = [] # children nodes
    
    
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
    

class MCTS:
    def __init__(self, sm):
        self.sm = sm
        self.root = Node()
        self.wins = {'player0':0, 'player1':0}
        self.num_rollouts = 0

    def loop(self, time_limit):
        timer = True
        start = time.time()
        run = 0
        while timer:
            print('----------------------------------')
            print("run")
            print(run)
            sm = deepcopy(self.sm)
            node, sm = self.select(sm)
            if(self.expand(node, sm)):
                node = random.choice(node.get_children())
                sm.move(node.get_action())
                if sm.winner is not None:
                    winner = sm.winner
                else:
                    winner = self.rollout(sm)
                self.backprop(node, winner)
            
            print('winner')
            print(sm.winner)
            if sm.winner == 0:
                self.wins['player0'] += 1
            if sm.winner == 1:
                self.wins['player1'] += 1


            current_time = time.time()
            if current_time - start > time_limit:
                
                timer = False
            
            run += 1

        print(node.get_children())

    def uct(self, node, child_node):
        return math.sqrt((math.log(float(node.get_num_visited())))/(1 + float(child_node.get_num_visited())))
    
    def select(self, sm):
        sm = sm
        node = self.root
        nodes_selected = 0
        while node.has_children():
            current = -1000000
            for child in node.get_children():
                temp = child.get_q() + self.uct(node, child)
                if temp > current:
                    best = child
                    current = temp
            node = best
            sm.move(node.get_action())
            nodes_selected += 1
        return node, sm
                

    def expand(self, node, sm):
        parent_node = node
        legal_actions = sm.get_possible_moves()
        if len(legal_actions) > 0:
            for action in legal_actions:
                node.children.append(Node(action, parent_node))
            return True
        else:
            return False

    def rollout(self, sm):
        self.num_rollouts += 1
        sm = sm
        while True:
            moves = sm.get_possible_moves()
            move = random.choice(moves)

            if sm.move(move):
                return sm.winner
   
    def backprop(self, node, winner):
        if winner == 0:
            outcome = 1
        else:
            outcome = -1

        while node.parent is not None:
            node.update_eval(outcome)
            node.update_n()
            node.update_q()
            node = node.parent
        
        node.update_n()
        

if __name__ == "__main__":
    sm = StateManager()
    mcts = MCTS(sm)
    mcts.loop(5)
    print(mcts.wins)
    print(mcts.num_rollouts)



