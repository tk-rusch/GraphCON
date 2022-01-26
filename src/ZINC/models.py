import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool

class GraphCON_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, dt=1., alpha=1., gamma=1.):
        super(GraphCON_GCN, self).__init__()
        self.nlayers = nlayers
        self.enc = nn.Linear(nfeat,nhid)
        self.res = nn.Linear(nhid,nhid)
        self.conv = GCNConv(nhid, nhid)
        self.dec = nn.Linear(nhid,nclass)
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma

    def res_connection(self, X):
        ## residual connection
        res = self.res(X) - self.conv.lin(X)
        return res

    def forward(self, data):
        ## the following encoder gives better results than using nn.embedding()
        input = data.x.float()
        edge_index = data.edge_index
        Y = self.enc(input)
        X = Y

        for i in range(self.nlayers):
            Y = Y + self.dt * (torch.relu(self.conv(X, edge_index) + self.res_connection(X)) - self.alpha * Y - self.gamma * X)
            X = X + self.dt * Y

        X = self.dec(X)
        X = global_add_pool(X, data.batch).squeeze(-1)

        return X