"""
Extracts the optimal hyperparameters found from a wandb hyperparameter sweep
"""

import yaml
import argparse
import wandb

import torch
import numpy as np

from good_params_waveGNN import good_params_dict
# from greed_params import default_params, not_sweep_args, greed_run_params
# from run_GNN import main
from data import get_dataset, set_train_val_test_split
from GNN import GNN
from GNN_early import GNNEarly
from run_GNN import get_optimizer, test, train


def main(opt, data_dir="../data"):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = get_dataset(opt, data_dir, opt['not_lcc'])

  if opt["num_splits"] > 0:
    dataset.data = set_train_val_test_split(
      23 * np.random.randint(0, opt["num_splits"]),
      # random prime 23 to make the splits 'more' random. Could remove
      dataset.data,
      num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)

  model = GNN(opt, dataset, device) if opt["no_early"] else GNNEarly(opt, dataset, device)
  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  model, data = model.to(device), dataset.data.to(device)
  parameters = [p for p in model.parameters() if p.requires_grad]
  optimizer = get_optimizer(opt["optimizer"], parameters, lr=opt["lr"], weight_decay=opt["decay"])

  # if checkpoint_dir:
  #   checkpoint = os.path.join(checkpoint_dir, "checkpoint")
  #   model_state, optimizer_state = torch.load(checkpoint)
  #   model.load_state_dict(model_state)
  #   optimizer.load_state_dict(optimizer_state)

  this_test = test
  best_time = best_epoch = train_acc = val_acc = test_acc = 0
  for epoch in range(1, opt["epoch"]):
    loss = train(model, optimizer, data)
    # need next line as it sets the attributes in the solver

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
    # with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
    #   path = os.path.join(checkpoint_dir, "checkpoint")
    #   torch.save((model.state_dict(), optimizer.state_dict()), path)
    # tune.report(loss=loss, accuracy=val_acc, test_acc=test_acc, train_acc=train_acc, best_time=best_time,
    #             best_epoch=best_epoch,
    #             forward_nfe=model.fm.sum, backward_nfe=model.bm.sum)
    wandb.log({"loss": loss,
               "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc, "best_time": best_time,
               "best_epoch": best_epoch, "epoch_step": epoch})


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
  for run in opt['runs']:
    # default_params_dict = default_params()
    # greed_run_dict = greed_run_params(default_params_dict)
    # not_sweep_dict = not_sweep_args(greed_run_dict, project_name, group_name)

    yaml_path = f"./wandb/sweep-{opt['sweep']}/config-{run}.yaml"
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

    opt = {**opt, **good_params_dict[dataset], **yaml_opt}
    opt['wandb'] = True
    opt['use_wandb_offline'] = False
    opt['wandb_best_run_id'] = run
    opt['wandb_track_grad_flow'] = False
    opt['wandb_watch_grad'] = False

    # for i in range(num_runs):
    main(opt)
  wandb_run.finish()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
  parser.add_argument('--sweep', type=str, default=None, help='sweep folder to read', required=True)
  parser.add_argument('--runs', type=str, nargs='+', default=None, help='the run IDs', required=True)
  parser.add_argument('--reps', type=int, default=1, help='the number of random weight initialisations to use')
  parser.add_argument('--name', type=str, default=None)
  # parser.add_argument('--function', type=str, default=None)
  # parser.add_argument('--block', type=str, default=None)
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

# if __name__ == "__main__":
#   sweep = 'ebq1b5hy'
#   run_list = ['yv3v42ym', '7ba0jl9m', 'a60dnqcc', 'v6ln1x90', 'f5dmv6ow']
#   project_name = 'best_runs'
#   group_name = 'eval'
#   run_best(sweep, run_list, project_name, group_name, num_runs=8)
