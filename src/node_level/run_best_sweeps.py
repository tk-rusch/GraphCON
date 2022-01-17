"""
Extracts the optimal hyperparameters found from a wandb hyperparameter sweep
"""

import yaml
import argparse
from good_params_waveGNN import good_params_dict
# from greed_params import default_params, not_sweep_args, greed_run_params
from run_GNN import main


def run_best(opt):
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

    opt = {**good_params_dict, **yaml_opt}
    opt['wandb_best_run_id'] = run

    # for i in range(num_runs):
    main(opt)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
  parser.add_argument('--sweep', type=str, default=None, help='sweep folder to read', required=True)
  parser.add_argument('--runs', type=str, nargs='+', default=None, help='the run IDs', required=True)
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

# if __name__ == "__main__":
#   sweep = 'ebq1b5hy'
#   run_list = ['yv3v42ym', '7ba0jl9m', 'a60dnqcc', 'v6ln1x90', 'f5dmv6ow']
#   project_name = 'best_runs'
#   group_name = 'eval'
#   run_best(sweep, run_list, project_name, group_name, num_runs=8)
