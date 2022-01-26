from torch_geometric.datasets import MNISTSuperpixels
DATA_PATH = '../../data'

def get_data(train):
  path = '../../data/MNIST'
  dataset = MNISTSuperpixels(path,train=train)
  return dataset