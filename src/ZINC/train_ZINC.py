from data_handling import get_zinc_data
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import *
from torch import nn
import os
from pathlib import Path
from torch_geometric.utils import degree
from sys import argv
from torch_geometric.nn import GCNConv, ChebConv
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def job_id_to_parameters():
    lr = np.logspace(-4, -2, 1000)
    N = int(np.random.randint(7,15))
    diff = np.linspace(0.1, 1.5, 1000)    
    freq = np.linspace(0.0, 1.5, 1000)

    return lr, N, diff, freq

def train_GNN(lr,N,diff,freq,id):
    train_dataset = get_zinc_data('train')
    test_dataset = get_zinc_data('test')
    val_dataset = get_zinc_data('val')

    train_loader = DataLoader(train_dataset, batch_size=128,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128,shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=128,shuffle=False)

    epochs = 2000
    hidden = 65

    reduce_point = 20
    patience = 0
    start_reduce = 1500
    best_eval = 1000000

    deg = torch.zeros(5, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    model = GraphCON_PNA(nfeat=train_dataset.data.num_features,
                     nhid=hidden,
                     nclass=1,
                     dropout=0,
                     N=N, dt=1., diffusion=diff, frequency=freq,deg=deg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lf = torch.nn.L1Loss()

    n_params = 0
    for param in model.parameters():
        n_params += param.numel()
    print(n_params)

    def test(loader):
        error = 0
        for data in loader:
            data = data.to(device)
            output = model(data)
            error += (output - data.y).abs().sum().item()
        return error / len(loader.dataset)

    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = lf(out,data.y)
            loss.backward()
            optimizer.step()

        val_loss = test(val_loader)
        test_loss = test(test_loader)

        if (val_loss < best_eval):
            best_eval = val_loss

        elif (val_loss >= best_eval and (epoch + 1) >= start_reduce):
            patience += 1

        if (epoch + 1) >= start_reduce and patience == reduce_point:
            patience = 0
            lr /= 2.
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            f = open('results_con_pna/id_' + str(id) + '.txt', 'a')
            f.write('reduced lr by factor of 2\n')
            f.close()

        if (epoch > 25):
            if (val_loss > 10. or lr < 1e-5):
                break

        Path('results_con_pna').mkdir(parents=True, exist_ok=True)
        f = open('results_con_pna/id_' + str(id) + '.txt', 'a')
        if (epoch == 0):
            f.write('## ' + str(lr) + ' ' + str(N) + ' ' + str(diff) + ' ' + str(freq) + '\n')
        f.write(str(test_loss) + ' ' + str(val_loss) + '\n')
        f.close()

        if (epoch > 25):
            if (val_loss > 20.):
                break

if __name__ == '__main__':
    id = int(argv[1])
    lr, N, diff, freq = job_id_to_parameters()
    train_GNN(lr, N, diff, freq, id)

