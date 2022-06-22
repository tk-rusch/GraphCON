from data_handling import get_data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from models import *
from torch_geometric.data import DataLoader
from best_params import best_params_dict
import argparse

def train_GNN(opt):
    train_dataset = get_data(train=True)
    test_dataset = get_data(train=False)

    train_dataset = train_dataset.shuffle()
    val_dataset = train_dataset[:5000]
    train_dataset = train_dataset[5000:]

    train_loader = DataLoader(train_dataset, batch_size=opt['batch'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt['batch'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=opt['batch'], shuffle=False)

    epochs = opt['epochs']

    if opt['model'] == 'GraphCON_GCN':
        model = GraphCON_GCN(nfeat=train_dataset.data.num_features+2,nhid=opt['nhid'],nclass=10,
                             dropout=opt['drop'],nlayers=opt['nlayers'],dt=1.,
                             alpha=opt['alpha'],gamma=opt['gamma']).to(opt['device'])
    elif opt['model'] == 'GraphCON_GAT':
        model = GraphCON_GAT(nfeat=train_dataset.data.num_features+2, nhid=opt['nhid'], nclass=10,
                             dropout=opt['drop'], nlayers=opt['nlayers'], dt=1.,
                             alpha=opt['alpha'], gamma=opt['gamma'],nheads=opt['nheads']).to(opt['device'])

    optimizer = optim.Adam(model.parameters(), lr=opt['lr'])
    lf = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=opt['reduce_factor'])

    best_eval = 0

    def test(data_loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                data = data.to(opt['device'])
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(data.y.data.view_as(pred)).sum()

        accuracy = 100. * correct / len(data_loader.dataset)
        return accuracy.item()

    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_loader):
            data = data.to(opt['device'])
            optimizer.zero_grad()
            out = model(data)
            loss = lf(out, data.y)
            loss.backward()
            optimizer.step()

        val_acc = test(val_loader)
        test_acc = test(test_loader)

        if(val_acc > best_eval):
            best_eval = val_acc
            best_test_acc = test_acc

        log = 'Epoch: {:03d}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, val_acc, test_acc))

        scheduler.step(val_acc)
        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']
        if(curr_lr<1e-5):
            break

        if(epoch > 25 and val_acc < 20.):
            break

    print('Final test accuracy: ', best_test_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--model', type=str, default='GraphCON_GAT',
                        help='GraphCON_GCN, GraphCON_GAT')
    parser.add_argument('--nhid', type=int, default=256,
                        help='number of hidden node features')
    parser.add_argument('--nlayers', type=int, default=5,
                        help='number of layers')
    parser.add_argument('--alpha', type=float, default=1.,
                        help='alpha parameter of graphCON')
    parser.add_argument('--gamma', type=float, default=1.,
                        help='gamma parameter of graphCON')
    parser.add_argument('--nheads', type=int, default=4,
                        help='number of attention heads for GraphCON-GAT')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='max epochs')
    parser.add_argument('--batch', type=int, default=32,
                        help='batch size')
    parser.add_argument('--reduce_factor', type=float, default=0.5,
                        help='reduce factor')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--drop', type=float, default=0.3,
                        help='dropout rate')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='computing device')

    args = parser.parse_args()
    cmd_opt = vars(args)

    best_opt = best_params_dict[cmd_opt['model']]
    opt = {**cmd_opt, **best_opt}
    print(opt)

    train_GNN(opt)

