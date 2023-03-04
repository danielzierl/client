from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.cluster import KMeans
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
        return (torch.from_numpy(np.asarray(tuple[0])),torch.from_numpy(np.asarray(tuple[1])))

device = torch.device('cpu')
lr= 0.002
num_of_epochs = 500
batch_size = 200
dataset = ClassificationFaceDataset(transform=toTensor())
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# model = model.NeuralNet(input_size=dataset.dataX.shape[1], hidden_size=1000, num_of_classes=2, device=device, training=False)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()
print(dataset.dataX.shape[1])
loss_fcn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.9)


def train(model):
    for epoch in range(num_of_epochs):
        for i, (dataX, labels) in enumerate(dataloader):
            labels = labels.type(torch.LongTensor).to(device)
            dataX = dataX.to(device)
            dataX = dataX.to(torch.float32)
            output = model(dataX).to(device)
            loss = loss_fcn(output, labels).to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%10==0:
                acc =evaluateAccuracy(model,dataset,test=True)
                print(f"epoch={epoch}, loss={loss}, test_accuracy={acc}, train_accuracy={evaluateAccuracy(model,dataset,test=False)}")
        step_lr_scheduler.step()

def evaluateAccuracy(model,dataset:ClassificationFaceDataset, test=False):
    corr=0
    for i, (dataX, labels) in enumerate(getCorrectAccuracyInputs(test)):
        labels = labels.type(torch.LongTensor).to(device)
        dataX = dataX.to(device)
        dataX = dataX.to(torch.float32)
        output =  model(dataX).to(device)
        _,predicted =torch.max(output, dim=1)
        if  test :
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


train(model)
