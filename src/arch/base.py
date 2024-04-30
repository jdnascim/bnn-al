import torch
import pyro.distributions as dist
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class BaseMLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_hidden_layers, linear=True, dropout=0.5):
        super(BaseMLP, self).__init__()

        self.linear = linear

        layers_size = []
        layers_size.append(input_size)
        for i in range(n_hidden_layers):
            layers_size.append(hidden_size)
        layers_size.append(output_size)

        self.layers = torch.nn.ModuleList()

        for i in range(len(layers_size)-1):
            if i < len(layers_size) - 2:
                self.layers.append(nn.Linear(layers_size[i], layers_size[i+1]))
            else:
                self.layers.append(nn.Linear(layers_size[i], layers_size[i+1]))
            
        self.dropout = dropout

    def forward(self, x):
        x = x.float()

        for l in self.layers:
            x = l(x)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()


class BaseGNN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_hidden_layers, linear=True, dropout=0.5):
        super(BaseGNN, self).__init__()

        self.linear = linear

        layers_size = []
        layers_size.append(input_size)
        for i in range(n_hidden_layers+1):
            layers_size.append(hidden_size)
        layers_size.append(output_size)

        self.layers = torch.nn.ModuleList()

        for i in range(len(layers_size)-1):
            if i < len(layers_size) - 2:
                self.layers.append(SAGEConv(layers_size[i], layers_size[i+1]))
            else:
                self.layers.append(nn.Linear(layers_size[i], layers_size[i+1]))
            
        self.dropout = dropout

    def forward(self, x, edge_index=None, dropout_infer=False):
        x = x.float()

        qtde_layers = len(self.layers)

        dropout_flag = (self.training or dropout_infer)
        for ix, l in enumerate(self.layers):

            if ix < len(self.layers) - 1:
                x = l(x, edge_index)
    
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=dropout_flag)
            elif self.linear is False:
                x = l(x, edge_index)
            else:
                x = l(x)
                

        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()