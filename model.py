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
