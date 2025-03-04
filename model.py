import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def _init_(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self)._init()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(input_size, hidden_size)
        self.l3 = nn.Linear(input_size, num_classes)
        self.relu = nn.Relu()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)

        # no activation and no softmax
        return out