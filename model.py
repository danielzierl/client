import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(torch.nn.Module):
    def __init__(self, input_size,hidden_size, num_of_classes,device,training=True) -> None:
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size,device=device)
        self.l2 = nn.Linear(hidden_size, hidden_size,device=device)
        self.l3 = nn.Linear(hidden_size, num_of_classes,device=device)
        self.device=device
        self.training =training
        self.input_size=input_size

    def forward(self,x):
        out = x.view(-1,self.input_size)
        out = nn.functional.dropout(nn.functional.relu(self.l1(out)))
        out = nn.functional.dropout( nn.functional.relu(self.l2(out)), p=0.5)
        out = self.l3(out)
        return out


class MobNet(torch.nn.Module):
    def __init__(self, input_size,hidden_size, num_of_classes,device,training=True) -> None:
        super(NeuralNet,self).__init__()

class CustomMobileNet(torch.nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(CustomMobileNet, self).__init__()
        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.top = nn.Linear(in_features=input_size, out_features=224*224*3)
        self.mid = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        for param in self.mid.parameters():
            param.requires_grad = False
        self.bottom = nn.Linear(1000, output_size)

    def forward(self,x):
        out = self.flatten(x)
        out = self.top(out)
        out = out.view(-1,3,224,224)
        out = self.mid(out)
        out = self.bottom(out)
        return torch.max(out, dim=0)


