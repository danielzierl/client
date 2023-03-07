from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
# import model
import torch.nn as nn
import copy
import os
from model import CustomMobileNet

save_path = "saved_model"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
data_file = "./data/data.csv"



def load_data(file, train_rate=0.8):
    data = np.loadtxt(file, delimiter=",", dtype=float)
    indexes = range(len(data)-1)

    y = torch.from_numpy(data[:, -1])
    data = data[:, :-1]

    y = nn.functional.one_hot(y.to(torch.int64))

    train_idx, test_idx = train_test_split(indexes, train_size=train_rate, random_state=43, shuffle=True)

    X_train, X_test = data[train_idx], data[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test

torch_model = CustomMobileNet(1404, 4).to(device)


if __name__ == "__main__":

    class CustomDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X).type(torch.float32).to(device)
            self.y = torch.tensor(y).type(torch.float32)
            self.y = self.y[:, None].to(device)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            x_value = self.X[idx, :]
            y_value = self.y[idx, :]
            return x_value, y_value


    


    def cross_entropy_one_hot(input, target):
        _, labels = target.max(dim=0)
        return nn.CrossEntropyLoss()(input, labels)

    batch_size = 20


    # Hyperparameters
    learning_rate = 0.001
    epochs = 20

    X_train, X_val, y_train, y_val = load_data(data_file, 0.2)
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    # Initialize the loss function
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
    # step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.9)


    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            y = torch.reshape(y, (-1, y.size()[-1]))
            pred = model(X)
            loss = loss_fn(pred, y)
            # loss = torch.sum(torch.eq(pred, y))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"Loss: {loss:>7f}, [{current:>5d}/{size:>5d}]")


    def test_loop(dataloader, model, loss_fn, print_it=True):
        
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                y = torch.reshape(y, (-1, y.size()[-1]))
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        if print_it:
            print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct

    # if os.path.isfile(save_path):
    #         torch_model.load_state_dict(torch.load(save_path))
    for t in range(epochs):
        
        # if t == 50:
        #     optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate2)
        print(f"Epoch {t + 1}\n---------------------------------")
        train_loop(train_dataloader, torch_model, loss_fn, optimizer)
        test_loop(val_dataloader, torch_model, loss_fn)
        torch.save(torch_model.state_dict(),save_path)
    print("Done! ")
