from torch_geometric.datasets import WebKB
import torch
import numpy as np

DATA_PATH = '../../data'

def get_data(name, split=0):
  path = '../../data/'+name
  dataset = WebKB(path,name=name)
  
  data = dataset[0]
  splits_file = np.load(f'{path}/{name}/raw/{name}_split_0.6_0.2_{split}.npz')
  train_mask = splits_file['train_mask']
  val_mask = splits_file['val_mask']
  test_mask = splits_file['test_mask']

  data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
  data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
  data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

  return data
