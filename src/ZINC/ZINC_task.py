from data_handling import get_zinc_data
import numpy as np
import torch
import torch.optim as optim
from models import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader

import argparse

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--nhid', type=int, default=220,
                    help='number of hidden node features')
parser.add_argument('--nlayers', type=int, default=22,
                    help='number of layers')
parser.add_argument('--alpha', type=float, default=0.215,
                    help='alpha parameter of graphCON')
parser.add_argument('--gamma', type=float, default=1.115,
                    help='gamma parameter of graphCON')
parser.add_argument('--epochs', type=int, default=3000,
                    help='max epochs')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    help='computing device')
parser.add_argument('--batch', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.00159,
                    help='learning rate')
parser.add_argument('--reduce_point', type=int, default=20,
                    help='length of patience')
parser.add_argument('--start_reduce', type=int, default=1000,
                    help='epoch when to start reducing')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

args = parser.parse_args()
print(args)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)

train_dataset = get_zinc_data('train')
test_dataset = get_zinc_data('test')
val_dataset = get_zinc_data('val')

train_loader = DataLoader(train_dataset, batch_size=args.batch,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch,shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch,shuffle=False)

model = GraphCON_GCN(nfeat=train_dataset.data.num_features,nhid=args.nhid,nclass=1,
                     nlayers=args.nlayers, dt=1., alpha=args.alpha,
                     gamma=args.gamma).to(args.device)

nparams = 0
for p in model.parameters():
    nparams += p.numel()
print('number of parameters: ',nparams)

optimizer = optim.Adam(model.parameters(),lr=args.lr)
lf = torch.nn.L1Loss()

patience = 0
best_eval = 1000000

def test(loader):
    model.eval()
    error = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(args.device)
            output = model(data)
            error += (output - data.y).abs().sum().item()
    return error / len(loader.dataset)

for epoch in range(args.epochs):
    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(args.device)
        optimizer.zero_grad()
        out = model(data)
        loss = lf(out,data.y)
        loss.backward()
        optimizer.step()

    val_loss = test(val_loader)

    f = open('zinc_graphcon_gcn_log.txt', 'a')
    f.write('validation loss: ' + str(val_loss) + '\n')
    f.close()

    print('epoch: ',epoch,'validation loss: ',val_loss)

    if (val_loss < best_eval):
        best_eval = val_loss
        best_test_loss = test(test_loader)

    elif (val_loss >= best_eval and (epoch + 1) >= args.start_reduce):
        patience += 1

    if (epoch + 1) >= args.start_reduce and patience == args.reduce_point:
        patience = 0
        args.lr /= 2.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    if (epoch > 25):
        if (val_loss > 20. or args.lr < 1e-5):
            break

f = open('zinc_graphcon_gcn_log.txt', 'a')
f.write('final test loss: ' + str(round(best_test_loss, 2)) + '\n')
f.close()
print('final test loss: ',val_loss)