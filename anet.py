import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
import numpy as np
import matplotlib.pyplot as plt

class ANET():
    def __init__(self, input_size, layers, learning_rate=0.01, optimizer=optim.Adam):
        self.model = NN(input_size, layers, learning_rate, optimizer)
        self.learning_rate = learning_rate
        self.losses = []
        self.rbuf_board = []
        self.rbuf_actions = []
 
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
        loss_fn = nn.BCELoss()
        x_train, y_train = next(iter(self.dataloader))
        for epoch in range(num_epochs):
            x_train = x_train.to(device=self.model.device)
            y_train = y_train.to(device=self.model.device)
            y_pred = self.model(x_train)
            loss = loss_fn(y_pred, y_train)
            self.losses.append(loss.item())
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

    def predicted_action_probs(self, board, legal_moves):
        moves = self.model(torch.Tensor(np.array(board))).tolist()
        moves = [0 if i < 0 else i for i in moves]
        for i in range(len(moves)):
            moves[i] = moves[i]*legal_moves[i]
        summ = sum(moves)
        moves = [x/summ for x in moves]
        return moves
    
    def get_best_move(self, board, legal_moves):
        moves = self.model(torch.Tensor(np.array(board))).tolist()
        for i in range(len(moves)):
            moves[i] = moves[i]*legal_moves[i]
        move = moves.index(max(moves))
        return move


    def save_model(self, interval_number):
        torch.save(self.model.state_dict(), f"saved_models/model_game_{interval_number}.pt")

    def print_losses(self):
        plt.plot(self.losses)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title("Learning rate %f"%(self.learning_rate))
        plt.show()

class NN(nn.Module):
    def __init__(self, input_size, layers, learning_rate=0.01, optimizer=optim.Adam):
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