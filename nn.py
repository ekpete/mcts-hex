import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader, TensorDataset 
import numpy as np

x = np.array([(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 0, 0)])

y = np.array([(0.003, 0.023, 0.117, 0.145, 0.003, 0.094, 0.019, 0.046, 0.058, 0.21, 0.113, 0.096, 0.009, 0.003, 0.044, 0.017), (0.026446280991735537, 0.045454545454545456, 0.013223140495867768, 0.047107438016528926, 0.11735537190082644, 0.017355371900826446, 0.20743801652892563, 0.14049586776859505, 0.023140495867768594, 0, 0.06115702479338843, 0.09173553719008265, 0.08677685950413223, 0.017355371900826446, 0.0768595041322314, 0.02809917355371901), (0.019184652278177457, 0.007993605115907274, 0.023980815347721823, 0.011990407673860911, 0.010391686650679457, 0.010391686650679457, 0, 0.006394884092725819, 0.04476418864908074, 0, 0.7673860911270983, 0.013589128697042365, 0.01598721023181455, 0.015187849720223821, 0.04476418864908074, 0.007993605115907274), (0.05359877488514548, 0.09137314956610515, 0.07912200102092905, 0.1066870852475753, 0.05666156202143951, 0.06789178152118427, 0, 0.061766207248596224, 0.08984175599795814, 0, 0, 0.1082184788157223, 0.13374170495150586, 0.013272077590607452, 0.07963246554364471, 0.05819295558958652)])

tensor_x = torch.Tensor(x)
tensor_y = torch.Tensor(y)

dataset = TensorDataset(tensor_x, tensor_y)
dataloader = DataLoader(dataset)


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = (self.fc2(x))
        return x 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
input_size = 17 
num_classes = 16
learning_rate = 0.1
num_epochs = 40

model = NN(input_size=input_size, num_classes=num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

optimizers = {"Adam": optim.Adam(model.parameters(), lr=learning_rate), 
              "Adagrad": optim.Adagrad(model.parameters(), lr=learning_rate), 
              "SGD": optim.SGD(model.parameters(), lr=learning_rate), 
              "RMSprop": optim.RMSprop(model.parameters(), lr=learning_rate)}

optimizer = optimizers["Adam"]

losses = []
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    for x_train, y_train in dataloader:
        # Get data to cuda if possible
        x_train = x_train.to(device=device)
        y_train = y_train.to(device=device)
        # forward propagation
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        losses.append(loss.item())
        # zero previous gradients
        optimizer.zero_grad()
        # back-propagation
        loss.backward()
        # gradient descent or adam step
        optimizer.step()
torch.set_printoptions(sci_mode=False)
print("Testing:")
moves = model(torch.Tensor(np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))).tolist()
new_moves = []
for move in moves:
    if move <0:
        new_moves.append(0)
    else:
        new_moves.append(move)

print(new_moves)


import matplotlib.pyplot as plt
plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f"%(learning_rate))
plt.show()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy"
            f" {float(num_correct) / float(num_samples) * 100:.2f}"
        )

    model.train()

#check_accuracy(dataloader, model)
