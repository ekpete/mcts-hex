import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
import numpy as np
import matplotlib.pyplot as plt
import math

class ANET():
    def __init__(self, input_size, layers, learning_rate=0.01, optimizer=optim.Adam):
        self.model = NN(input_size, layers, learning_rate, optimizer)
        self.learning_rate = learning_rate
        self.losses = []
        self.rbuf_board = []
        self.rbuf_actions = []
        self.get_all_moves(int(math.sqrt(input_size-1)))
 
    def update_dataset(self, rbuf, batch_size=64):
        self.rbuf_board = rbuf['board']
        self.rbuf_actions = rbuf['action_probs']
        tensor_x = torch.Tensor(np.array(self.rbuf_board))
        tensor_y = torch.Tensor(np.array(self.rbuf_actions))
        self.dataset = TensorDataset(tensor_x, tensor_y)
        self.dataloader = DataLoader(self.dataset, shuffle=True, batch_size=batch_size)
      
    def train(self, rbuf, batch_size):
        num_epochs = 1
        self.update_dataset(rbuf, batch_size)
        loss_fn = nn.CrossEntropyLoss()
        x_train, y_train = next(iter(self.dataloader))
        for epoch in range(num_epochs):
            x_train = x_train.to(device=self.model.device)
            y_train = y_train.to(device=self.model.device)
            y_pred = self.model(x_train)
            # print("-------")
            # print(x_train[0])
            # print(y_train[0])
            # print(y_pred[0])
            # print(loss_fn(y_pred, y_train))
            loss = loss_fn(y_pred, y_train)
            self.losses.append(loss.item())
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

    def predicted_action_probs(self, board, legal_moves):
        moves = self.model(torch.Tensor(np.array(board))).tolist()
        for i in range(len(moves)):
            moves[i] = moves[i]*legal_moves[i]
        summ = sum(moves)
        moves = [x/summ for x in moves]
        return moves
    
    def get_move(self, board):
        moves = nn.Softmax(dim=0)(self.model(torch.Tensor(np.array(board)))).tolist()
        legal_moves = []
        for pos in board[1:]:
            legal_moves.append(1) if int(pos)==0 else legal_moves.append(0)
        for i in range(len(moves)):
            moves[i] = moves[i]*legal_moves[i]
        return self.all_moves[moves.index(max(moves))]
        summ = sum(moves)
        moves = [x/summ for x in moves]
        return self.all_moves[moves.index(max(moves))]
    
    def get_all_moves(self, k):
        self.all_moves = []
        for i in range(k):
            for j in range(k):
                self.all_moves.append((i,j))

    def save_model(self, interval_number):
        torch.save(self.model.state_dict(), f"saved_models/actor_{interval_number}.pt")

    def print_losses(self):
        plt.plot(self.losses)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title("Learning rate %f"%(self.learning_rate))
        plt.show()

class TOPP_agent():
    def __init__(self, board_size, PATH):
        self.model = NN(input_size=(board_size**2)+1)
        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()
        self.name = PATH[13:-3]
        self.get_all_moves(board_size)
    
    def get_move(self, board):
        moves = nn.Softmax(dim=0)(self.model(torch.Tensor(np.array(board)))).tolist()
        legal_moves = []
        for pos in board[1:]:
            legal_moves.append(1) if int(pos)==0 else legal_moves.append(0)
        for i in range(len(moves)):
            moves[i] = moves[i]*legal_moves[i]
        summ = sum(moves)
        moves = [x/summ for x in moves]
        return self.all_moves[moves.index(max(moves))]
    
    def get_all_moves(self, k):
        self.all_moves = []
        for i in range(k):
            for j in range(k):
                self.all_moves.append((i,j))

class NN(nn.Module):
    def __init__(self, input_size, layers=[(25, nn.ReLU()),(25, None)], learning_rate=0.01, optimizer=optim.Adam):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.layers = nn.ModuleList()
        for size, activation in layers:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size
            if activation is not None:
                self.layers.append(activation)
        self.learning_rate = learning_rate
        self.optimizer = optimizer(params=self.parameters(), lr=learning_rate)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    """

    model = NN(26, [(30, nn.ReLU()), (25, None)])
    model.load_state_dict(torch.load("saved_models/model_game_20.pt"))
    model.eval()

    board1 = (2, 1, 2, 1, 2, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 0, 0, 0, 0)
    legal = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1)

    moves = model(torch.Tensor(np.array(board1))).tolist()
    print(moves)
    """

    #board1 = (1,1,1,2,0,0,2,1,2,1,0,2,1,1,2,0,2,0,0,0,0,0,0,0,0,0)
    #t0 = TOPP_agent(5, "saved_models/model_game_0.pt")
    #print(t0.get_move(board1))

    t = torch.Tensor([0,0,0,1,1,-3,-5,-2,0,1,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1])
    t2  = torch.Tensor([0.0274, 0.0137, 0.0076, 0.0274, 0.0655, 0.0076, 0.0076, 0.0396, 0.0137,
        0.0473, 0.0655, 0.0686, 0.0137, 0.0076, 0.0442, 0.0351, 0.0000, 0.1113,
        0.0137, 0.0549, 0.0000, 0.0000, 0.2652, 0.0442, 0.0183])
    t3 = torch.Tensor([-0.3251, -0.0478, -0.1883, -0.1321, 0.5158, 0.0390, -0.0827, 0.4843, 0.9044,
        0.1550, -0.0356, 0.0850, 0.5414, 0.1250, -0.6865, 0.6674, -0.5220, -0.1984,
        0.0137, 0.0549, 0.0000, 0.0000, 0.2652, 0.0442, 0.0183])
    print(nn.CrossEntropyLoss()(t3,t2))

