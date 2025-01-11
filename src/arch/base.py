import torch
import pyro.distributions as dist
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class BaseMLP(torch.nn.Module):
    def __init__(self, **kwargs):
        super(BaseMLP, self).__init__()

        input_dim = kwargs.get("input_dim")
        hidden_dim = kwargs.get("hidden_dim")
        output_dim = kwargs.get("output_dim")

        self.aug_unlbl_set = kwargs.get("aug_unlbl_set")
        self.dataset_loss = kwargs.get("dataset_loss")
        self.dropout = kwargs.get("dropout")

        if self.aug_unlbl_set is not None:
            if self.dataset_loss is True:
                output_dim += 2
            else:
                output_dim += 1

        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index=None, num_inference=None):
        x = x.float()

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.fc(x)

        if self.aug_unlbl_set is not None and self.dataset_loss is True:
            x[:, :2] = F.log_softmax(x[:, :2], dim=1)
            x[:, 2:] = F.log_softmax(x[:, 2:], dim=1)
        else:
            x = F.log_softmax(x, dim=1)

        if self.aug_unlbl_set is not None and self.training is False:
            positive_column = x[:, 1].unsqueeze(-1)  
            max_negative = torch.max(x[:, 0], x[:, 2]).unsqueeze(-1)  
            x = torch.cat((max_negative, positive_column), dim=1)  

        return x
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc.reset_parameters()


class BaseGNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(BaseGNN, self).__init__()

        input_dim = kwargs.get("input_dim")
        hidden_dim = kwargs.get("hidden_dim")
        output_dim = kwargs.get("output_dim")

        self.aug_unlbl_set = kwargs.get("aug_unlbl_set")
        self.dataset_loss = kwargs.get("dataset_loss")
        self.dropout = kwargs.get("dropout")

        if self.aug_unlbl_set is not None:
            if self.dataset_loss is True:
                output_dim += 2
            else:
                output_dim += 1

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index=None, num_inference=None):
        x = x.float()

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)

        if self.aug_unlbl_set is not None and self.dataset_loss is True:
            x[:, :2] = F.log_softmax(x[:, :2], dim=1)
            x[:, 2:] = F.log_softmax(x[:, 2:], dim=1)
        else:
            x = F.log_softmax(x, dim=1)

        if self.aug_unlbl_set is not None and self.training is False:
            positive_column = x[:, 1].unsqueeze(-1)  
            max_negative = torch.max(x[:, 0], x[:, 2]).unsqueeze(-1)  
            x = torch.cat((max_negative, positive_column), dim=1)  

        return x
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc.reset_parameters()