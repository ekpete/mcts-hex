import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
import numpy as np
import matplotlib.pyplot as plt

class ANET():
    def __init__(self, input_size, layers, learning_rate=0.1, optimizer=optim.Adam):
        self.model = NN(input_size, layers, learning_rate, optimizer)
        self.learning_rate = learning_rate
        self.losses = []
        self.rbuf_board = []
        self.rbuf_actions = []

    def update_dataset(self, rbuf_board, rbuf_actions, batch_size=4):
        self.rbuf_board.extend(rbuf_board)
        self.rbuf_actions.extend(rbuf_actions)
        tensor_x = torch.Tensor(np.array(self.rbuf_board))
        tensor_y = torch.Tensor(np.array(self.rbuf_actions))
        self.dataset = TensorDataset(tensor_x, tensor_y)
        self.dataloader = DataLoader(self.dataset, shuffle=True, batch_size=batch_size)

            
    def train(self, num_epochs, rbuf_board, rbuf_actions, batch_size):
        self.update_dataset(rbuf_board, rbuf_actions, batch_size)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            for x_train, y_train in self.dataloader:
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

    def save_model(self, interval_number):
        torch.save(self.model.state_dict(), f"saved_models/model_game_{interval_number}.pt")

    def print_losses(self):
        plt.plot(self.losses)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title("Learning rate %f"%(self.learning_rate))
        plt.show()

class NN(nn.Module):
    def __init__(self, input_size, layers, learning_rate=0.1, optimizer=optim.Adam):
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
    x = [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 0, 0)]

    y = [(0.003, 0.023, 0.117, 0.145, 0.003, 0.094, 0.019, 0.046, 0.058, 0.21, 0.113, 0.096, 0.009, 0.003, 0.044, 0.017), (0.026446280991735537, 0.045454545454545456, 0.013223140495867768, 0.047107438016528926, 0.11735537190082644, 0.017355371900826446, 0.20743801652892563, 0.14049586776859505, 0.023140495867768594, 0, 0.06115702479338843, 0.09173553719008265, 0.08677685950413223, 0.017355371900826446, 0.0768595041322314, 0.02809917355371901), (0.019184652278177457, 0.007993605115907274, 0.023980815347721823, 0.011990407673860911, 0.010391686650679457, 0.010391686650679457, 0, 0.006394884092725819, 0.04476418864908074, 0, 0.7673860911270983, 0.013589128697042365, 0.01598721023181455, 0.015187849720223821, 0.04476418864908074, 0.007993605115907274), (0.05359877488514548, 0.09137314956610515, 0.07912200102092905, 0.1066870852475753, 0.05666156202143951, 0.06789178152118427, 0, 0.061766207248596224, 0.08984175599795814, 0, 0, 0.1082184788157223, 0.13374170495150586, 0.013272077590607452, 0.07963246554364471, 0.05819295558958652)]

    input_size = 17 
    dense_size = 30
    output_size = 16

    learning_rate = 0.1
    num_epochs = 40
    layers=[(dense_size, nn.ReLU()), (output_size, None)]

    anet = ANET(input_size=input_size, layers=layers, learning_rate=learning_rate)
    anet.train(num_epochs, x, y, 4)
    board1 = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    legal = (0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0)
    moves = anet.predicted_action_probs(board1, legal)
    #anet.save_model(50)
    print(moves)
    anet.print_losses()