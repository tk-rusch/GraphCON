"""
Extracts the optimal hyperparameters found from a wandb hyperparameter sweep
"""

import yaml
import argparse
import wandb

import torch
import numpy as np
from ray import tune
from functools import partial
import time
from ray.tune import CLIReporter

from good_params_waveGNN import good_params_dict
from data import get_dataset, set_train_val_test_split
from GNN import GNN
from GNN_early import GNNEarly
from run_GNN import get_optimizer, test, train
from utils import get_sem, mean_confidence_interval


def main(opt, data_dir="../data"):
  # todo see if I can initialise wandb runs inside of ray processes
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = get_dataset(opt, data_dir, opt['not_lcc'])

  if opt["num_splits"] > 0:
    dataset.data = set_train_val_test_split(
      23 * np.random.randint(0, opt["num_splits"]),
      # random prime 23 to make the splits 'more' random. Could remove
      dataset.data,
      num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)

  model = GNN(opt, dataset, device) if opt["no_early"] else GNNEarly(opt, dataset, device)
  model, data = model.to(device), dataset.data.to(device)
  parameters = [p for p in model.parameters() if p.requires_grad]
  optimizer = get_optimizer(opt["optimizer"], parameters, lr=opt["lr"], weight_decay=opt["decay"])

  this_test = test
  best_time = best_epoch = train_acc = val_acc = test_acc = 0
  for epoch in range(1, opt["epoch"]):
    loss = train(model, optimizer, data)
    if opt["no_early"]:
      tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, opt)
      best_time = opt['time']
    else:
      tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, opt)
    if tmp_val_acc > val_acc:
      best_epoch = epoch
      train_acc = tmp_train_acc
      val_acc = tmp_val_acc
      test_acc = tmp_test_acc
    if model.odeblock.test_integrator.solver.best_val > val_acc:
      best_epoch = epoch
      val_acc = model.odeblock.test_integrator.solver.best_val
      test_acc = model.odeblock.test_integrator.solver.best_test
      train_acc = model.odeblock.test_integrator.solver.best_train
      best_time = model.odeblock.test_integrator.solver.best_time
    tune.report(loss=loss, val_acc=val_acc, test_acc=test_acc, train_acc=train_acc, best_time=best_time,
                best_epoch=best_epoch,
                forward_nfe=model.fm.sum, backward_nfe=model.bm.sum)
    res_dict = {"loss": loss,
                "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc, "best_time": best_time,
                "best_epoch": best_epoch, "epoch_step": epoch}
    print(res_dict)


def run_best(opt):
  opt['wandb_entity'] = 'bchamberlain'
  opt['wandb_project'] = 'waveGNN-src_node_level'
  opt['wandb_group'] = None
  if 'wandb_run_name' in opt.keys():
    wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                           name=opt['wandb_run_name'], reinit=True, config=opt)
  else:
    wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                           reinit=True, config=opt)

  wandb.define_metric("epoch_step")

  yaml_path = f"./wandb/sweep-{opt['sweep']}/config-{opt['run']}.yaml"
  with open(yaml_path) as f:
    yaml_opt = yaml.load(f, Loader=yaml.FullLoader)
  temp_opt = {}
  for k, v in yaml_opt.items():
    if type(v) == dict:
      temp_opt[k] = v['value']
    else:
      temp_opt[k] = v
  yaml_opt = temp_opt
  dataset = yaml_opt['dataset']

  opt = {**good_params_dict[dataset], **yaml_opt, **opt}
  opt['wandb'] = True
  opt['use_wandb_offline'] = False
  opt['wandb_best_run_id'] = opt['run']
  opt['wandb_track_grad_flow'] = False
  opt['wandb_watch_grad'] = False

  reporter = CLIReporter(
    metric_columns=["val_acc", "loss", "test_acc", "train_acc", "best_time", "best_epoch"])

  result = tune.run(
    partial(main, data_dir="../data"),
    name=run,
    resources_per_trial={"cpu": opt['cpus'], "gpu": opt['gpus']},
    search_alg=None,
    keep_checkpoints_num=3,
    checkpoint_score_attr='val_acc',
    config=opt,
    num_samples=opt['reps'] if opt["num_splits"] == 0 else opt["num_splits"] * opt["reps"],
    scheduler=None,
    max_failures=1,  # early stop solver can't recover from failure as it doesn't own m2.
    local_dir='../ray_tune',
    progress_reporter=reporter,
    raise_on_failed_trial=False)

  df = result.dataframe(metric='test_acc', mode="max").sort_values('test_acc', ascending=False)
  try:
    df.to_csv('../ray_results/{}_{}.csv'.format(run, time.strftime("%Y%m%d-%H%M%S")))
  except:
    pass

  print(df[['val_acc', 'test_acc', 'train_acc', 'best_time', 'best_epoch']])

  test_accs = df['test_acc'].values
  val_accs = df['val_acc'].values
  train_accs = df['train_acc'].values
  print("test accuracy {}".format(test_accs))
  log = "mean test {:04f}, test std {:04f}, test sem {:04f}, test 95% conf {:04f}"
  log_dic = {'test_acc': test_accs.mean(), 'val_acc': val_accs.mean(), 'train_acc': train_accs.mean(),
             'test_std': np.std(test_accs), 'test_sem': get_sem(test_accs),
             'test_95_conf': mean_confidence_interval(test_accs)}
  wandb.log(log_dic)
  print(log.format(test_accs.mean(), np.std(test_accs), get_sem(test_accs), mean_confidence_interval(test_accs)))

  wandb_run.finish()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
  parser.add_argument('--sweep', type=str, default=None, help='sweep folder to read', required=True)
  parser.add_argument('--run', type=str, default=None, help='the run IDs', required=True)
  parser.add_argument('--reps', type=int, default=1, help='the number of random weight initialisations to use')
  parser.add_argument('--name', type=str, default=None)
  parser.add_argument('--gpus', type=float, default=0, help='number of gpus per trial. Can be fractional')
  parser.add_argument('--cpus', type=float, default=1, help='number of cpus per trial. Can be fractional')
  parser.add_argument("--num_splits", type=int, default=0, help="Number of random slpits >= 0. 0 for planetoid split")
  parser.add_argument("--adjoint", dest='adjoint', action='store_true',
                      help="use the adjoint ODE method to reduce memory footprint")
  parser.add_argument("--max_nfe", type=int, default=5000,
                      help="Maximum number of function evaluations allowed in an epcoh.")
  parser.add_argument("--no_early", action="store_true",
                      help="Whether or not to use early stopping of the ODE integrator when testing.")

  parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')

  args = parser.parse_args()

  opt = vars(args)
  run_best(opt)
