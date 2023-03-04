import math

from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch
# import model
import torch.nn as nn
import copy


class ClassificationFaceDataset(Dataset):
    test_percentage = 0.75

    def __init__(self, transform=None) -> None:
        
        self.dataX = np.loadtxt('./data/data.csv', delimiter=",", dtype=float)
        self.total_count = self.dataX.shape[0]
        np.random.shuffle(self.dataX)
        data_count = self.dataX.shape[0]
        train_data_count = round(data_count * self.test_percentage)
        test_data_count = self.total_count-train_data_count
        self.labels = copy.deepcopy(self.dataX[:train_data_count,-1])
        self.testY = self.dataX[train_data_count:,-1]

        self.testX = self.dataX[train_data_count:,:-1]
        self.dataX = copy.deepcopy(self.dataX[:train_data_count,:-1])
    
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = (self.dataX[index], self.labels[index])
        if self.transform: item=self.transform(item)
        return item

class ClassificationEasyDataset(Dataset):

    def __init__(self, transform=None) -> None:
        self.dataX = np.loadtxt('./data/test.csv', delimiter=",", dtype=int)
        
        self.labels = self.dataX[:,-1]
        self.dataX = self.dataX[:,:-1]
        print(self.labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = (self.dataX[index], self.labels[index])
        if self.transform: item=self.transform(item)
        return item

class toTensor():
    def __call__(self, tuple) -> any:
        return (torch.from_numpy(np.asarray(tuple[0])), torch.from_numpy(np.asarray(tuple[1])))

device = torch.device('cuda')
# lr= 0.0002
# num_of_epochs = 500
# batch_size = 10
# dataset = ClassificationFaceDataset(transform=toTensor())
# dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# model = model.CustomMobileNet(dataset.dataX.shape[1], 2)
# model = model.to(device)

# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# model.eval()
# print(dataset.dataX.shape[1])
# loss_fcn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(),lr=lr)
# step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.9)



# def train(model):
#     for epoch in range(num_of_epochs):
#         for i, (dataX, labels) in enumerate(dataloader):
#             labels = labels.type(torch.LongTensor).to(device)
#             dataX = dataX.to(device).to(torch.float32)
#             output = model(dataX)
#             loss = loss_fcn(output, labels).to(device)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             if i%10==0:
#                 acc =evaluateAccuracy(model,dataset,test=True)
#                 print(f"epoch={epoch}, loss={loss}, test_accuracy={acc}, train_accuracy={evaluateAccuracy(model,dataset,test=False)}")
#         step_lr_scheduler.step()

# def evaluateAccuracy(model,dataset:ClassificationFaceDataset, test=False):
#     corr = 0
#     for i, (dataX, labels) in enumerate(getCorrectAccuracyInputs(test)):
#         labels = labels.type(torch.LongTensor).to(device)
#         dataX = dataX.to(device).to(torch.float32)
#         output = model(dataX).to(device)
#         _,predicted =torch.max(output, dim=1)
#         if test:
#             # print(labels)
#             # print(dataX[:3])
#             # print(predicted)
#             pass
#         corr += (predicted[0]).eq(labels).sum()
#     acc = corr /(dataset.total_count*(1-dataset.test_percentage if test else dataset.test_percentage))
#     return acc.item()


# def getCorrectAccuracyInputs(test=False):
#     if test:
#         return zip(torch.from_numpy(dataset.testX), torch.from_numpy(dataset.testY))
#     else:
#         return dataset

def load_data(file, num_cls, rate=0.8):
    data = np.loadtxt(file, delimiter=",", dtype=float)
    X = data[:, :-2]
    y = data[:, -1]

    size = len(data)

    train, val = torch.utils.data.random_split(data, [math.ceil(size*rate), size-math.ceil(size*rate)])
    X_train = X[:math.ceil(size*rate), :]
    X_test = X[:math.ceil(size*rate), :]

    y_train = torch.from_numpy(y[:math.ceil(size * rate)])
    y_test = torch.from_numpy(y[math.ceil(size * rate):])

    # y_train = torch.reshape(y_train, (-1, 1))
    # y_test = torch.reshape(y_test, (-1, 1))

    y_train = nn.functional.one_hot(y_train.to(torch.int64))
    y_test = nn.functional.one_hot(y_test.to(torch.int64))

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # train(model)

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


    class CustomMobileNet(torch.nn.Module):
        def __init__(self, input_size, output_size) -> None:
            super(CustomMobileNet, self).__init__()
            self.flatten = nn.Flatten()
            self.input_size = input_size
            self.top = nn.Linear(in_features=input_size, out_features=224 * 224 * 3)
            self.mid = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
            for param in self.mid.parameters():
                param.requires_grad = False
            self.bottom = nn.Linear(1000, output_size)
            self.soft = nn.Softmax()

        def forward(self, x):
            out = self.flatten(x)
            out = self.top(out)
            out = out.view(-1, 3, 224, 224)
            out = self.mid(out)
            out = self.bottom(out)
            # out = self.soft(out)
            return out


    def cross_entropy_one_hot(input, target):
        _, labels = target.max(dim=0)
        return nn.CrossEntropyLoss()(input, labels)

    batch_size = 20

    torch_model = CustomMobileNet(1403, 2).to(device)

    # Hyperparameters
    learning_rate = 0.02
    learning_rate2 = 0.2
    epochs = 20

    data_file = "./data/data.csv"
    X_train, X_val, y_train, y_val = load_data(data_file, 2)
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


    for t in range(epochs):
        # if t == 50:
        #     optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate2)
        print(f"Epoch {t + 1}\n---------------------------------")
        train_loop(train_dataloader, torch_model, loss_fn, optimizer)
        test_loop(val_dataloader, torch_model, loss_fn)
    print("Done!")