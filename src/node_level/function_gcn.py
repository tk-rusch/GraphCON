import torch
from torch import nn
import torch_sparse
import torch.nn.functional as F
from base_classes import ODEFunc
from utils import MaxNFEException
from torch_geometric.nn.conv import GCNConv


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class GCNFunc(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(GCNFunc, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))
    self.conv = GCNConv(in_features, out_features, normalize=True)

  def sparse_multiply(self, x):
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      # ax = torch.mean(torch.stack(
      #   [torch_sparse.spmm(self.edge_index, self.attention_weights[:, idx], x.shape[0], x.shape[0], x) for idx in
      #    range(self.opt['heads'])], dim=0), dim=0)
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x_full):  # the t param is needed by the ODE solver.
    x = x_full[:, :self.opt['hidden_dim']]
    y = x_full[:, self.opt['hidden_dim']:]
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    # ay = self.sparse_multiply(y)
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train
    f = self.conv(y, self.edge_index) - x - y
    # f = (ay - x - y)
    if self.opt['add_source']:
      f = (1. - F.sigmoid(self.beta_train)) * f + F.sigmoid(self.beta_train) * self.x0[:, self.opt['hidden_dim']:]
    f = torch.cat([f, (1. - F.sigmoid(self.beta_train2)) * alpha * x + F.sigmoid(self.beta_train2) *
                   self.x0[:, :self.opt['hidden_dim']]], dim=1)
    return f
