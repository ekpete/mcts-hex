import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
import numpy as np

x = np.array([(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 0, 0)])

y = np.array([(0.003, 0.023, 0.117, 0.145, 0.003, 0.094, 0.019, 0.046, 0.058, 0.21, 0.113, 0.096, 0.009, 0.003, 0.044, 0.017), (0.026446280991735537, 0.045454545454545456, 0.013223140495867768, 0.047107438016528926, 0.11735537190082644, 0.017355371900826446, 0.20743801652892563, 0.14049586776859505, 0.023140495867768594, 0, 0.06115702479338843, 0.09173553719008265, 0.08677685950413223, 0.017355371900826446, 0.0768595041322314, 0.02809917355371901), (0.019184652278177457, 0.007993605115907274, 0.023980815347721823, 0.011990407673860911, 0.010391686650679457, 0.010391686650679457, 0, 0.006394884092725819, 0.04476418864908074, 0, 0.7673860911270983, 0.013589128697042365, 0.01598721023181455, 0.015187849720223821, 0.04476418864908074, 0.007993605115907274), (0.05359877488514548, 0.09137314956610515, 0.07912200102092905, 0.1066870852475753, 0.05666156202143951, 0.06789178152118427, 0, 0.061766207248596224, 0.08984175599795814, 0, 0, 0.1082184788157223, 0.13374170495150586, 0.013272077590607452, 0.07963246554364471, 0.05819295558958652)])

tensor_x = torch.Tensor(x)
tensor_y = torch.Tensor(y)

dataset = TensorDataset(tensor_x, tensor_y)
batch_size = 4
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

class NN(nn.Module):
    def __init__(self, input_size, layers_data, learning_rate=0.1, optimizer=optim.Adam):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.layers = nn.ModuleList()
        for size, activation in layers_data:
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

input_size = 17 
dense_size = 30
output_size = 16

learning_rate = 0.1
num_epochs = 40
layers_data=[(dense_size, nn.ReLU()), (output_size, None)]

model = NN(input_size=input_size, layers_data=layers_data, learning_rate=learning_rate, optimizer=optim.Adam)
loss_fn = nn.CrossEntropyLoss()

losses = []
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    for x_train, y_train in dataloader:
        x_train = x_train.to(device=model.device)
        y_train = y_train.to(device=model.device)
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        losses.append(loss.item())
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
    
torch.set_printoptions(sci_mode=False)
print("Testing:")
moves = model(torch.Tensor(np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))).tolist()
moves = [0 if i < 0 else i for i in moves]
print(moves)
print(model)

import matplotlib.pyplot as plt
plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f"%(learning_rate))
plt.show()