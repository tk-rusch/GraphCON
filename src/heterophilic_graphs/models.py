import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv, GATConv

class GraphCON_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers, dt=1., alpha=1., gamma=1., res_version=1):
        super(GraphCON_GCN, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nlayers = nlayers
        self.enc = nn.Linear(nfeat,nhid)
        self.conv = GCNConv(nhid, nhid)
        self.dec = nn.Linear(nhid,nclass)
        self.res = nn.Linear(nhid,nhid)
        if(res_version==1):
            self.residual = self.res_connection_v1
        else:
            self.residual = self.res_connection_v2
        self.dt = dt
        self.act_fn = nn.ReLU()
        self.alpha = alpha
        self.gamma = gamma
        self.reset_params()

    def reset_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'emb' not in name and 'out' not in name:
                stdv = 1. / math.sqrt(self.nhid)
                param.data.uniform_(-stdv, stdv)

    def res_connection_v1(self, X):
        res = - self.res(self.conv.lin(X))
        return res

    def res_connection_v2(self, X):
        res = - self.conv.lin(X) + self.res(X)
        return res

    def forward(self, data):
        input = data.x
        edge_index = data.edge_index
        input = F.dropout(input, self.dropout, training=self.training)
        Y = self.act_fn(self.enc(input))
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayers):
            Y = Y + self.dt*(self.act_fn(self.conv(X,edge_index) + self.residual(X)) - self.alpha*Y - self.gamma*X)
            X = X + self.dt*Y
            Y = F.dropout(Y, self.dropout, training=self.training)
            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)

        return X

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
        input = data.x
        n_nodes = input.size(0)
        edge_index = data.edge_index
        input = F.dropout(input, self.dropout, training=self.training)
        Y = self.act_fn(self.enc(input))
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayers):
            Y = Y + self.dt*(F.elu(self.conv(X, edge_index) + self.res_connection(X)).view(n_nodes, -1, self.nheads).mean(dim=-1) - self.alpha*Y - self.gamma*X)
            X = X + self.dt*Y
            Y = F.dropout(Y, self.dropout, training=self.training)
            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)

        return X
