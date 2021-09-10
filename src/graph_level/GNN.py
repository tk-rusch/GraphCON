import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from base_classes import BaseGNN
from model_configurations import set_block, set_function
from torch_geometric.nn import global_mean_pool, global_add_pool

class GNN(BaseGNN):
    def __init__(self, opt, dataset, device=torch.device('cpu')):
        super(GNN, self).__init__(opt, dataset, device)
        self.f = set_function(opt)
        self.embeddings = nn.Parameter(torch.ones((5, opt['hidden_dim'])))
        block = set_block(opt)
        time_tensor = torch.tensor([0, self.T]).to(device)
        self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)

    def forward(self, data):
        x = torch.index_select(self.embeddings, 0, data.x.long())
        pos = data.pos
        dist_l = (pos[data.edge_index[0]] - pos[data.edge_index[1]]).pow(2).sum(dim=-1).sqrt()
        self.odeblock.odefunc.edge_index = data.edge_index
        self.odeblock.odefunc.edge_weight = dist_l

        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        #x = self.m1(x)
        x = torch.cat([x, x], dim=-1)
        self.odeblock.set_x0(x)
        z = self.odeblock(x)
        z = z[:, self.opt['hidden_dim']:]
        # Activation.
        z = F.relu(z)
        # Dropout.
        z = F.dropout(z, self.opt['dropout'], training=self.training)
        # Decode each node embedding to get node label.
        z = self.m2(z)
        out = global_add_pool(z, data.batch)
        return out