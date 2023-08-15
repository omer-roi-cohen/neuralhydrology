import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_nse_cdf(metric_csv, path):
    metric_df = pd.read_csv(metric_csv)
    nse_vec = metric_df['NSE'].to_numpy()
    plot_ecdf(nse_vec, path)


def ecdf(vec):
    x, counts = np.unique(vec, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]


def plot_ecdf(vec, path,param='',value=0):
    x, y = ecdf(vec[vec > 0])
    x = np.insert(x, 0, x[0])
    y = np.insert(y, 0, 0.)
    fig = plt.figure(figsize=(10, 12))
    plt.plot(x, y)
    plt.grid(True)
    param_str = f'{param}={value}' if param else ''
    plt.title(f'NSE CDF \n({np.round(100 * len(vec[vec < 0]) / len(vec),1)}% of NSEs < 0)\n{param_str}')
    plt.xlabel('NSE')
    plt.ylabel('CDF')
    fig.savefig(path)
    plt.show()

def get_nse_ecdf_vec(path):
  metric_df = pd.read_csv(path)
  nse_vec = metric_df['NSE'].to_numpy()
  return nse_vec

def create_dict(params_and_val, run_names):
  vecs_dict = {}
  i = 1
  for p_v, run_name in zip(params_and_val, run_names):
      vecs_dict[f'{i}. ' + p_v[0]] = [p_v[1],get_nse_ecdf_vec(run_name)]
      i += 1
  return vecs_dict

def cdf_xy_to_plot(vec):
  x, y = ecdf(vec[vec > 0])
  x = np.insert(x, 0, x[0])
  y = np.insert(y, 0, 0.)
  return x, y

def plot_many_cdfs(vecs_dict,compared_vec):
  fig = plt.figure(figsize=(12, 10))
  for param in vecs_dict:
    vec = vecs_dict[param][1]
    x, y = cdf_xy_to_plot(vec)
    plt.plot(x, y, label=f'{param}={vecs_dict[param][0]}')
  x_c, y_c = cdf_xy_to_plot(compared_vec)
  plt.plot(x_c, y_c, label=f'Benchmark',linestyle='--', linewidth=1.5,c='black')
  plt.grid(True)
  plt.title(f'NSE CDF of au by parameter values')
  plt.xlabel('NSE')
  plt.ylabel('CDF')
  plt.legend()
  fig.savefig('cdf_by_params.png')
  plt.show()

