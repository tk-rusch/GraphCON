import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool

class GraphCON_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers, dt=1., alpha=1., gamma=1.):
        super(GraphCON_GCN, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nlayers = nlayers
        self.enc = nn.Linear(nfeat,nhid)
        self.conv = GCNConv(nhid, nhid)
        self.dec = nn.Linear(nhid,nclass)
        self.res = nn.Linear(nhid,nhid)
        self.dt = dt
        self.act_fn = nn.ReLU()
        self.alpha = alpha
        self.gamma = gamma

    def res_connection(self, X):
        res = self.res(X)
        return res

    def forward(self, data):
        input = torch.cat([data.x,data.pos],dim=-1)
        edge_index = data.edge_index
        Y = self.enc(input)
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayers):
            Y = Y + self.dt*(self.act_fn(self.conv(X,edge_index) + self.res_connection(X)) - self.alpha*Y - self.gamma*X)
            X = X + self.dt*Y
            Y = F.dropout(Y, self.dropout, training=self.training)
            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)
        X = global_add_pool(X, data.batch)

        return X.squeeze(-1)

class GraphCON_GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, dropout, dt=1., alpha=1., gamma=1., nheads=4):
        super(GraphCON_GAT, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dropout = dropout
        self.nheads = nheads
        self.nhid = nhid
        self.nlayers = nlayers
        self.act_fn = nn.ReLU()
        self.res = nn.Linear(nhid, nheads * nhid)
        self.enc = nn.Linear(nfeat,nhid)
        self.conv = GATConv(nhid, nhid, heads=nheads)
        self.dec = nn.Linear(nhid,nclass)
        self.dt = dt

    def res_connection(self, X):
        res = self.res(X)
        return res

    def forward(self, data):
        input = torch.cat([data.x, data.pos], dim=-1)
        n_nodes = input.size(0)
        edge_index = data.edge_index
        Y = self.enc(input)
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayers):
            Y = Y + self.dt*(F.elu(self.conv(X, edge_index) + self.res_connection(X)).view(n_nodes, -1, self.nheads).mean(dim=-1) - self.alpha*Y - self.gamma*X)
            X = X + self.dt*Y
            Y = F.dropout(Y, self.dropout, training=self.training)
            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)
        X = global_add_pool(X, data.batch)

        return X.squeeze(-1)