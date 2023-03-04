import math

from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import model
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
lr= 0.0002
num_of_epochs = 500
batch_size = 10
dataset = ClassificationFaceDataset(transform=toTensor())
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=2)

model = model.CustomMobileNet(dataset.dataX.shape[1], 2)
model = model.to(device)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()
print(dataset.dataX.shape[1])
loss_fcn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=lr)
step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.9)



def train(model):
    for epoch in range(num_of_epochs):
        for i, (dataX, labels) in enumerate(dataloader):
            labels = labels.type(torch.LongTensor).to(device)
            dataX = dataX.to(device).to(torch.float32)
            output = model(dataX)
            loss = loss_fcn(output, labels).to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%10==0:
                acc =evaluateAccuracy(model,dataset,test=True)
                print(f"epoch={epoch}, loss={loss}, test_accuracy={acc}, train_accuracy={evaluateAccuracy(model,dataset,test=False)}")
        step_lr_scheduler.step()

def evaluateAccuracy(model,dataset:ClassificationFaceDataset, test=False):
    corr = 0
    for i, (dataX, labels) in enumerate(getCorrectAccuracyInputs(test)):
        labels = labels.type(torch.LongTensor).to(device)
        dataX = dataX.to(device).to(torch.float32)
        output = model(dataX).to(device)
        _,predicted =torch.max(output, dim=1)
        if test:
            # print(labels)
            # print(dataX[:3])
            # print(predicted)
            pass
        corr += (predicted[0]).eq(labels).sum()
    acc = corr /(dataset.total_count*(1-dataset.test_percentage if test else dataset.test_percentage))
    return acc.item()


def getCorrectAccuracyInputs(test=False):
    if test:
        return zip(torch.from_numpy(dataset.testX), torch.from_numpy(dataset.testY))
    else:
        return dataset

def load_data(file, num_cls, train_rate=0.8):
    data = np.loadtxt(file, delimiter=",", dtype=float)

    data_train, data_val = train_test_split(data, train_size=train_rate, random_state=43, shuffle=True)

    X_train, y_train = data_train[:, :-2], data_train[:, -1]
    X_test, y_test = data_val[:, :-2], data_val[:, -1]
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    y_train = torch.reshape(y_train, (-1, 1))
    y_test = torch.reshape(y_test, (-1, 1))

    y_train = nn.functional.one_hot(y_train.to(torch.int64))
    y_test = nn.functional.one_hot(y_test.to(torch.int64))

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train()
