import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import SpGraphAttentionLayer
from torch.autograd import Variable
from torch.nn import ModuleList
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GCNConv, GATConv, PNAConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_sort_pool

class GraphCON_GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, N, dt=1., diffusion=1., frequency=1.,nheads=4):
        super(GraphCON_GAT, self).__init__()
        self.dropout = dropout
        self.N = N
        self.nheads = nheads
        self.emb = nn.Linear(nfeat,nhid)
        self.res = nn.Linear(nhid,nheads*nhid)
        self.attentions = [SpGraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=0.2, concat=True, residual=True)
                           for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_h_{}'.format(i), attention)
        self.out = nn.Linear(nhid,nclass)
        self.dt = dt
        self.diffusion = diffusion
        self.frequency = frequency

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        x = self.emb(x)
        y = x

        for i in range(self.N):
            #print(self.conv(y,edge_index).view(n_nodes,-1,self.nheads).mean(dim=-1).size(),x.size())
            x = x + self.dt * (torch.cat([att(x, edge_index).unsqueeze(-1) for att in self.attentions], dim=-1).mean(dim=-1) - self.diffusion * x - self.frequency * y)
            y = y + self.dt * x

        y = self.out(y)
        y = global_add_pool(y, data.batch)

        return y.squeeze(-1)

class GraphCON(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, N, dt=1., diffusion=1., frequency=1.):
        super(GraphCON, self).__init__()
        self.dropout = dropout
        self.N = N
        self.emb = nn.Linear(nfeat,nhid)
        self.res = nn.Linear(nhid,nhid)
        self.conv = GCNConv(nhid, nhid)
        self.out = nn.Linear(nhid,nclass)
        self.dt = dt
        self.diffusion = diffusion
        self.frequency = frequency

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        x = self.emb(x)
        y = x

        for i in range(self.N):
            x = x + self.dt * (torch.relu(self.conv(y, edge_index) - self.conv.lin(y) + self.res(y)) - self.diffusion * x - self.frequency * y)
            y = y + self.dt * x

        y = self.out(y)
        y = global_add_pool(y, data.batch)

        return y.squeeze(-1)

class GraphCON_PNA(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, N, dt=1., diffusion=1., frequency=1., deg=1):
        super(GraphCON_PNA, self).__init__()
        self.dropout = dropout
        self.N = N
        self.emb = nn.Linear(nfeat,nhid)
        self.res = nn.Linear(nhid,nhid)
        #self.batch_norms = ModuleList()
        #for i in range(N):
        #    self.batch_norms.append(BatchNorm(nhid))
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.conv = PNAConv(in_channels=nhid, out_channels=nhid,
                       aggregators=aggregators, scalers=scalers, deg=deg, towers=5, pre_layers=1, post_layers=1,
                       divide_input=False)

        self.out = nn.Linear(nhid,nclass)
        self.dt = dt
        self.diffusion = diffusion
        self.frequency = frequency

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        x = self.emb(x)
        y = x

        for i in range(self.N):
            #x_new = self.batch_norms[i](torch.relu(self.conv(y, edge_index) + self.res(y)))
            x = x + self.dt * (torch.relu(self.conv(y, edge_index) + self.res(y)) - self.diffusion * x - self.frequency * y)
            y = y + self.dt * x

        y = self.out(x)
        y = global_add_pool(y, data.batch)

        return y.squeeze(-1)