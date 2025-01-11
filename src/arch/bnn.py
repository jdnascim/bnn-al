import pyro
import torch
import pyro.distributions as dist
import torch.nn as nn
from pyro.nn import PyroModule, PyroSample
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

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
        self.dataset_loss = kwargs.get("dataset_loss")

        if self.aug_unlbl_set is not None:
            if self.dataset_loss is True:
                output_dim += 2
            else:
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

        if self.aug_unlbl_set is not None and self.dataset_loss is True:
            x[:, :2] = F.log_softmax(x[:, :2], dim=1)
            x[:, 2:] = F.log_softmax(x[:, 2:], dim=1)
        else:
            x = F.log_softmax(x, dim=1)

        return x

    # Returns B x n x output
    def forward(self, input_B: torch.Tensor, edge_index: torch.Tensor, k=1):
        self.k = k
        
        mc_input_BK = self.mc_tensor(input_B, self.k)
        edge_index_BK = self.mc_tensor_edge(edge_index, self.k)
        mc_output_BK = self.mc_forward_impl(mc_input_BK, edge_index_BK, self.k)
        mc_output_B_K = self.unflatten_tensor(mc_output_BK, self.k)

        if self.aug_unlbl_set is not None and self.training is False:
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

        if self.aug_unlbl_set is not None:
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
        edge_index_BK = self.mc_tensor_edge(edge_index, self.k)

        mc_output_BK = self.mc_forward_impl(mc_input_BK, edge_index_BK, self.k)
        mc_output_B_K = self.unflatten_tensor(mc_output_BK, self.k)
    
        if self.aug_unlbl_set is not None and self.training is False:
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
        self.dataset_loss = kwargs.get("dataset_loss")

        if self.aug_unlbl_set is not None:
            if self.dataset_loss is True:
                output_dim += 2
            else:
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

        if self.aug_unlbl_set is not None and self.dataset_loss is True:
            x[:, :2] = F.log_softmax(x[:, :2], dim=1)
            x[:, 2:] = F.log_softmax(x[:, 2:], dim=1)
        else:
            x = F.log_softmax(x, dim=1)

        return x

    # Returns B x n x output
    def forward(self, input_B: torch.Tensor, edge_index: torch.Tensor, k=1):
        self.k = k
        
        mc_input_BK = self.mc_tensor(input_B, self.k)
        mc_output_BK = self.mc_forward_impl(mc_input_BK, self.k)
        mc_output_B_K = self.unflatten_tensor(mc_output_BK, self.k)
        
        if self.aug_unlbl_set is not None and self.training is False:
            positive_column = mc_output_B_K[:, :, 1].unsqueeze(-1)  
            max_negative = torch.max(mc_output_B_K[:, :, 0], mc_output_B_K[:, :, 2]).unsqueeze(-1)  
            mc_output_B_K = torch.cat((max_negative, positive_column), dim=2)  

        return mc_output_B_K
    
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


class BayesianHeteroGNN(consistent_mc_dropout.BayesianModule):
    
    def __init__(self, **kwargs):
        super().__init__()

        hidden_dim = kwargs.get("hidden_dim")
        output_dim = kwargs.get("output_dim")

        self.aug_unlbl_set = kwargs.get("aug_unlbl_set")
        self.dataset_loss = kwargs.get("dataset_loss")

        self.single_modal_loss = kwargs.get("single_modal_loss")

        if self.aug_unlbl_set is not None:
            if self.dataset_loss is True:
                output_dim += 2
            else:
                output_dim += 1

        self.conv1 = HeteroConv({
            ('image', 'sim', 'image'): SAGEConv((-1,-1), hidden_dim),
            ('text', 'sim', 'text'): SAGEConv((-1,-1), hidden_dim),
            ('joint', 'sim', 'joint'): SAGEConv((-1,-1), hidden_dim),
            ('image', 'ref', 'joint'): SAGEConv((-1, -1), hidden_dim),
            ('text', 'ref', 'joint'): SAGEConv((-1, -1), hidden_dim)
        }, aggr='sum')
        self.conv1_drop = ConsistentMCDropout()
        
        self.conv2 = HeteroConv({
            ('image', 'sim', 'image'): SAGEConv(hidden_dim, hidden_dim),
            ('text', 'sim', 'text'): SAGEConv(hidden_dim, hidden_dim),
            ('joint', 'sim', 'joint'): SAGEConv(hidden_dim, hidden_dim),
            ('image', 'ref', 'joint'): SAGEConv(hidden_dim, hidden_dim),
            ('text', 'ref', 'joint'): SAGEConv(hidden_dim, hidden_dim)
        }, aggr='sum')
        self.conv2_drop = ConsistentMCDropout()

        if self.single_modal_loss is not False:
            self.fc_image = nn.Linear(hidden_dim, output_dim)
            self.fc_text = nn.Linear(hidden_dim, output_dim)

            if self.single_modal_loss == "late":
                self.fc_joint = nn.Linear(hidden_dim, output_dim)
                self.fc = nn.Linear(3*output_dim, output_dim)
            elif self.single_modal_loss == "middle":
                self.fc = nn.Linear(3*hidden_dim, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def mc_forward_impl(self, x_dict, edge_index_dict, k):
        x_dict = {key: F.relu(self.conv1_drop(x, k), inplace=True) for key, x in self.conv1(x_dict, edge_index_dict).items()}
        x_dict = {key: F.relu(self.conv2_drop(x, k), inplace=True) for key, x in self.conv2(x_dict, edge_index_dict).items()}

        if self.single_modal_loss is not False:
            x = [None, None, None, None]
            x_trans = [None, None, None, None]
            for i_m, modal in enumerate(["joint", "image", "text"]):

                x[i_m] = x_dict[modal]
                x[i_m] = self.fc_image(x[i_m])
        
                if self.aug_unlbl_set is not None and self.dataset_loss is True:
                    x[i_m][:, :2] = F.log_softmax(x[i_m][:, :2], dim=1)
                    x[i_m][:, 2:] = F.log_softmax(x[i_m][:, 2:], dim=1)
                else:
                    x[i_m] = F.log_softmax(x[i_m], dim=1)   

                # for posterior concatenation, order image and text indices based on joint indices
                if modal in ("image", "text"):
                    x_trans[i_m] = x[i_m][edge_index_dict[(modal, "ref", "joint")][0]]
                else:
                    x_trans[i_m] = x[i_m]
                
            if self.single_modal_loss == "late":
                result_tensor = torch.cat(x_trans[:-1], dim=1)
                result_tensor = self.fc(result_tensor)
            elif self.single_modal_loss == "middle":
                x_image = x_dict["image"][edge_index_dict[("image", "ref", "joint")][0]]
                x_text = x_dict["text"][edge_index_dict[("text", "ref", "joint")][0]]
                result_tensor = torch.cat([x_image, x_text, x_dict["joint"]], dim=1)
                result_tensor = self.fc(result_tensor)
            
            x[-1] = result_tensor
                
            if self.aug_unlbl_set is not None and self.dataset_loss is True:
                x[i_m][:, :2] = F.log_softmax(x[i_m][:, :2], dim=1)
                x[i_m][:, 2:] = F.log_softmax(x[i_m][:, 2:], dim=1)
            else:
                x[i_m] = F.log_softmax(x[i_m], dim=1)   

        else:
            joint_x = x_dict['joint']
            x = self.fc(joint_x)
    
            if self.aug_unlbl_set is not None and self.dataset_loss is True:
                x[:, :2] = F.log_softmax(x[:, :2], dim=1)
                x[:, 2:] = F.log_softmax(x[:, 2:], dim=1)
            else:
                x = F.log_softmax(x, dim=1)

        return x

    def forward(self, x_dict, edge_index_dict, k=1):
        self.k = k
        
        mc_x_dict_BK = {key: self.mc_tensor(x, self.k) for key, x in x_dict.items()}
        mc_edge_index_dict_BK = {key: self.mc_tensor_edge(edge_index, self.k) for key, edge_index in edge_index_dict.items()}

        if self.single_modal_loss is not False:
            mc_output_BK_vec = self.mc_forward_impl(mc_x_dict_BK, mc_edge_index_dict_BK, self.k)
            mc_output_B_K = [None, None, None, None]

            for i, mc_output_BK in enumerate(mc_output_BK_vec):
                mc_output_B_K[i] = self.unflatten_tensor(mc_output_BK, self.k)
        
                if self.aug_unlbl_set is not None and not self.training:
                    positive_column = mc_output_B_K[i][:, :, 1].unsqueeze(-1)  
                    max_negative = torch.max(mc_output_B_K[i][:, :, 0], mc_output_B_K[i][:, :, 2]).unsqueeze(-1)  
                    mc_output_B_K[i] = torch.cat((max_negative, positive_column), dim=2)  
        else:
            mc_output_BK = self.mc_forward_impl(mc_x_dict_BK, edge_index_dict, self.k)
            mc_output_B_K = self.unflatten_tensor(mc_output_BK, self.k)
    
            if self.aug_unlbl_set is not None and not self.training:
                positive_column = mc_output_B_K[:, :, 1].unsqueeze(-1)  
                max_negative = torch.max(mc_output_B_K[:, :, 0], mc_output_B_K[:, :, 2]).unsqueeze(-1)  
                mc_output_B_K = torch.cat((max_negative, positive_column), dim=2)  

        return mc_output_B_K
    
    def reset_parameters(self):
        for conv in [self.conv1, self.conv2]:
            for key in conv.convs:
                conv.convs[key].reset_parameters()
        self.fc.reset_parameters()

class BayesianLateGNN(consistent_mc_dropout.BayesianModule):
    
    def __init__(self, **kwargs):
        super().__init__()

        hidden_dim = kwargs.get("hidden_dim")
        output_dim = kwargs.get("output_dim")

        self.aug_unlbl_set = kwargs.get("aug_unlbl_set")
        self.dataset_loss = kwargs.get("dataset_loss")

        self.single_modal_loss = kwargs.get("single_modal_loss")

        if self.aug_unlbl_set is not None:
            if self.dataset_loss is True:
                output_dim += 2
            else:
                output_dim += 1

        self.conv1 = HeteroConv({
            ('image', 'sim', 'image'): SAGEConv((-1,-1), hidden_dim),
            ('text', 'sim', 'text'): SAGEConv((-1,-1), hidden_dim),
            ('joint', 'sim', 'joint'): SAGEConv((-1,-1), hidden_dim),
            ('image', 'ref', 'joint'): SAGEConv((-1, -1), hidden_dim),
            ('text', 'ref', 'joint'): SAGEConv((-1, -1), hidden_dim)
        }, aggr='sum')
        self.conv1_drop = ConsistentMCDropout()
        
        self.conv2 = HeteroConv({
            ('image', 'sim', 'image'): SAGEConv(hidden_dim, hidden_dim),
            ('text', 'sim', 'text'): SAGEConv(hidden_dim, hidden_dim),
            ('joint', 'sim', 'joint'): SAGEConv(hidden_dim, hidden_dim),
            ('image', 'ref', 'joint'): SAGEConv(hidden_dim, hidden_dim),
            ('text', 'ref', 'joint'): SAGEConv(hidden_dim, hidden_dim)
        }, aggr='sum')
        self.conv2_drop = ConsistentMCDropout()

        if self.single_modal_loss is not False:
            self.fc_image = nn.Linear(hidden_dim, output_dim)
            self.fc_text = nn.Linear(hidden_dim, output_dim)

            if self.single_modal_loss == "late":
                self.fc_joint = nn.Linear(hidden_dim, output_dim)
                self.fc = nn.Linear(3*output_dim, output_dim)
            elif self.single_modal_loss == "middle":
                self.fc = nn.Linear(3*hidden_dim, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def mc_forward_impl(self, x_dict, edge_index_dict, k):
        x_dict = {key: F.relu(self.conv1_drop(x, k), inplace=True) for key, x in self.conv1(x_dict, edge_index_dict).items()}
        x_dict = {key: F.relu(self.conv2_drop(x, k), inplace=True) for key, x in self.conv2(x_dict, edge_index_dict).items()}

        if self.single_modal_loss is not False:
            x = [None, None, None, None]
            x_trans = [None, None, None, None]
            for i_m, modal in enumerate(["joint", "image", "text"]):

                x[i_m] = x_dict[modal]
                x[i_m] = self.fc_image(x[i_m])
        
                if self.aug_unlbl_set is not None and self.dataset_loss is True:
                    x[i_m][:, :2] = F.log_softmax(x[i_m][:, :2], dim=1)
                    x[i_m][:, 2:] = F.log_softmax(x[i_m][:, 2:], dim=1)
                else:
                    x[i_m] = F.log_softmax(x[i_m], dim=1)   

                # for posterior concatenation, order image and text indices based on joint indices
                if modal in ("image", "text"):
                    x_trans[i_m] = x[i_m][edge_index_dict[(modal, "ref", "joint")][0]]
                else:
                    x_trans[i_m] = x[i_m]
                
            if self.single_modal_loss == "late":
                result_tensor = torch.cat(x_trans[:-1], dim=1)
                result_tensor = self.fc(result_tensor)
            elif self.single_modal_loss == "middle":
                x_image = x_dict["image"][edge_index_dict[("image", "ref", "joint")][0]]
                x_text = x_dict["text"][edge_index_dict[("text", "ref", "joint")][0]]
                result_tensor = torch.cat([x_image, x_text, x_dict["joint"]], dim=1)
                result_tensor = self.fc(result_tensor)
            
            x[-1] = result_tensor
                
            if self.aug_unlbl_set is not None and self.dataset_loss is True:
                x[i_m][:, :2] = F.log_softmax(x[i_m][:, :2], dim=1)
                x[i_m][:, 2:] = F.log_softmax(x[i_m][:, 2:], dim=1)
            else:
                x[i_m] = F.log_softmax(x[i_m], dim=1)   

        else:
            joint_x = x_dict['joint']
            x = self.fc(joint_x)
    
            if self.aug_unlbl_set is not None and self.dataset_loss is True:
                x[:, :2] = F.log_softmax(x[:, :2], dim=1)
                x[:, 2:] = F.log_softmax(x[:, 2:], dim=1)
            else:
                x = F.log_softmax(x, dim=1)

        return x

    def forward(self, x_dict, edge_index_dict, k=1):
        self.k = k
        
        mc_x_dict_BK = {key: self.mc_tensor(x, self.k) for key, x in x_dict.items()}
        mc_edge_index_dict_BK = {key: self.mc_tensor_edge(edge_index, self.k) for key, edge_index in edge_index_dict.items()}

        if self.single_modal_loss is not False:
            mc_output_BK_vec = self.mc_forward_impl(mc_x_dict_BK, mc_edge_index_dict_BK, self.k)
            mc_output_B_K = [None, None, None, None]

            for i, mc_output_BK in enumerate(mc_output_BK_vec):
                mc_output_B_K[i] = self.unflatten_tensor(mc_output_BK, self.k)
        
                if self.aug_unlbl_set is not None and not self.training:
                    positive_column = mc_output_B_K[i][:, :, 1].unsqueeze(-1)  
                    max_negative = torch.max(mc_output_B_K[i][:, :, 0], mc_output_B_K[i][:, :, 2]).unsqueeze(-1)  
                    mc_output_B_K[i] = torch.cat((max_negative, positive_column), dim=2)  
        else:
            mc_output_BK = self.mc_forward_impl(mc_x_dict_BK, edge_index_dict, self.k)
            mc_output_B_K = self.unflatten_tensor(mc_output_BK, self.k)
    
            if self.aug_unlbl_set is not None and not self.training:
                positive_column = mc_output_B_K[:, :, 1].unsqueeze(-1)  
                max_negative = torch.max(mc_output_B_K[:, :, 0], mc_output_B_K[:, :, 2]).unsqueeze(-1)  
                mc_output_B_K = torch.cat((max_negative, positive_column), dim=2)  

        return mc_output_B_K
    
    def reset_parameters(self):
        for conv in [self.conv1, self.conv2]:
            for key in conv.convs:
                conv.convs[key].reset_parameters()
        self.fc.reset_parameters()
