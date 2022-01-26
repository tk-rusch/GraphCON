import os
from torch_geometric.datasets import ZINC
DATA_PATH = '../data'

def get_zinc_data(split):
  path = '../data/ZINC'
  dataset = ZINC(path,subset=True,split=split)
  return dataset