import pyro
import torch
import pyro.distributions as dist
import torch.nn as nn
from pyro.nn import PyroModule, PyroSample
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from batchbald_redux import (
    active_learning,
    batchbald,
    consistent_mc_dropout,
    joint_entropy,
    repeated_mnist,
)

class ConsistentMCDropout(consistent_mc_dropout._ConsistentMCDropout):
    r"""Randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. The elements to zero are randomized on every forward call during training time.

    During eval time, a fixed mask is picked and kept until `reset_mask()` is called.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .

    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: `Any`. Input can be of any shape
        - Output: `Same`. Output is of the same shape as input

    Examples::

        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """
    def __init__(self):
        super().__init__()

        self.p = 0.2

    def forward(self, input: torch.Tensor, k):
        if self.p == 0.0:
            return input

        if self.training:
            # Create a new mask on each call and for each batch element.
            k = input.shape[0]
            mask = self._create_mask(input, k)
        else:
            if self.mask is None:
                # print('recreating mask', self)
                # Recreate mask.
                self.mask = self._create_mask(input, k)

            mask = self.mask

        mc_input = consistent_mc_dropout.BayesianModule.unflatten_tensor(input, k)
        mc_output = mc_input.masked_fill(mask, 0) / (1 - self.p)

        # Flatten MCDI, batch into one dimension again.
        return consistent_mc_dropout.BayesianModule.flatten_tensor(mc_output)

class BayesianGNN(consistent_mc_dropout.BayesianModule):
    
    def __init__(self, **kwargs):
        super().__init__()

        input_dim = kwargs.get("input_dim")
        hidden_dim = kwargs.get("hidden_dim")
        output_dim = kwargs.get("output_dim")

        self.aug_unlbl_set = kwargs.get("aug_unlbl_set")

        if self.aug_unlbl_set is True:
            output_dim += 1

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv1_drop = ConsistentMCDropout()
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2_drop = ConsistentMCDropout()
        self.fc = nn.Linear(hidden_dim, output_dim)


    def mc_forward_impl(self, x: torch.Tensor, edge_index: torch.Tensor, k):
        x = F.relu(self.conv1_drop(self.conv1(x, edge_index), k), 2)
        x = F.relu(self.conv2_drop(self.conv2(x, edge_index), k), 2)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x

    # Returns B x n x output
    def forward(self, input_B: torch.Tensor, edge_index: torch.Tensor, k=1):
        self.k = k
        
        mc_input_BK = self.mc_tensor(input_B, self.k)
        mc_output_BK = self.mc_forward_impl(mc_input_BK, edge_index, self.k)
        mc_output_B_K = self.unflatten_tensor(mc_output_BK, self.k)

        if self.aug_unlbl_set is True and self.training is False:
            positive_column = mc_output_B_K[:, :, 1].unsqueeze(-1)  
            max_negative = torch.max(mc_output_B_K[:, :, 0], mc_output_B_K[:, :, 2]).unsqueeze(-1)  
            mc_output_B_K = torch.cat((max_negative, positive_column), dim=2)  

        return mc_output_B_K
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc.reset_parameters()

class BayesianHybrid(consistent_mc_dropout.BayesianModule):
    
    def __init__(self, **kwargs):
        super().__init__()

        input_dim = kwargs.get("input_dim")
        hidden_dim = kwargs.get("hidden_dim")
        output_dim = kwargs.get("output_dim")

        self.aug_unlbl_set = kwargs.get("aug_unlbl_set")

        if self.aug_unlbl_set is True:
            output_dim += 1

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv1_drop = ConsistentMCDropout()
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.conv2_drop = ConsistentMCDropout()
        self.fc = nn.Linear(hidden_dim, output_dim)


    def mc_forward_impl(self, x: torch.Tensor, edge_index: torch.Tensor, k):
        x = F.relu(self.conv1_drop(self.conv1(x, edge_index), k), 2)
        x = F.relu(self.conv2_drop(self.lin(x), k), 2)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x

    # Returns B x n x output
    def forward(self, input_B: torch.Tensor, edge_index: torch.Tensor, k=1):
        self.k = k
        
        mc_input_BK = self.mc_tensor(input_B, self.k)
        mc_output_BK = self.mc_forward_impl(mc_input_BK, edge_index, self.k)
        mc_output_B_K = self.unflatten_tensor(mc_output_BK, self.k)
    
        if self.aug_unlbl_set is True and self.training is False:
            positive_column = mc_output_B_K[:, :, 1].unsqueeze(-1)  
            max_negative = torch.max(mc_output_B_K[:, :, 0], mc_output_B_K[:, :, 2]).unsqueeze(-1)  
            mc_output_B_K = torch.cat((max_negative, positive_column), dim=2)  

        return mc_output_B_K
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.lin.reset_parameters()
        self.fc.reset_parameters()

class BayesianMLP(consistent_mc_dropout.BayesianModule):
    
    def __init__(self, **kwargs):
        super().__init__()

        input_dim = kwargs.get("input_dim")
        hidden_dim = kwargs.get("hidden_dim")
        output_dim = kwargs.get("output_dim")
        
        self.aug_unlbl_set = kwargs.get("aug_unlbl_set")

        if self.aug_unlbl_set is True:
            output_dim += 1

        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.conv1_drop = ConsistentMCDropout()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.conv2_drop = ConsistentMCDropout()
        self.fc = nn.Linear(hidden_dim, output_dim)


    def mc_forward_impl(self, x: torch.Tensor, k):
        x = F.relu(self.conv1_drop(self.lin1(x), k), 2)
        x = F.relu(self.conv2_drop(self.lin2(x), k), 2)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x

    # Returns B x n x output
    def forward(self, input_B: torch.Tensor, edge_index: torch.Tensor, k=1):
        self.k = k
        
        mc_input_BK = self.mc_tensor(input_B, self.k)
        mc_output_BK = self.mc_forward_impl(mc_input_BK, self.k)
        mc_output_B_K = self.unflatten_tensor(mc_output_BK, self.k)
        
        if self.aug_unlbl_set is True and self.training is False:
            positive_column = mc_output_B_K[:, :, 1].unsqueeze(-1)  
            max_negative = torch.max(mc_output_B_K[:, :, 0], mc_output_B_K[:, :, 2]).unsqueeze(-1)  
            mc_output_B_K = torch.cat((max_negative, positive_column), dim=2)  

        return mc_output_B_K
    
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()