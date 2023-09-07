from captum.attr import IntegratedGradients
import torch
import numpy as np
import datetime
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader

from neuralhydrology.datasetzoo import get_dataset, camelsus
from neuralhydrology.evaluation.utils import load_scaler
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.customlstm import CustomLSTM
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config

def get_date_range_and_idx(start_date, end_date, date_range):
  start_date_pd = pd.to_datetime(datetime.datetime(start_date[0], start_date[1], start_date[2], 0, 0))
  end_date_pd = pd.to_datetime(datetime.datetime(end_date[0], end_date[1], end_date[2], 0, 0))
  idx  = np.where(np.bitwise_and(start_date_pd <= date_range, date_range <= end_date_pd))[0]
  date_range_out = pd.date_range(start_date_pd, end_date_pd)
  return date_range_out, idx

config_file = Path(r'Configs\integrated_gradients_test1.yml')
cudalstm_config = Config(config_file)
run_dir = Path(r'runs\cudalstm_train_cluster_60_train_2108_112051')  # maybe get from config?

# Load model with random weights
cuda_lstm = CudaLSTM(cfg=cudalstm_config)
# load the trained weights into the new model.
model_path = run_dir / 'model_epoch022.pt'
model_weights = torch.load(str(model_path), map_location='cuda:0')  # load the weights from the file, creating the weight tensors on CPU
cuda_lstm.load_state_dict(model_weights)  # set the new model's weights to the values loaded from file
cuda_lstm.eval()


# load the dataset
scaler = load_scaler(run_dir)
dataset = get_dataset(cudalstm_config, is_train=False, period='test', scaler=scaler)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, collate_fn=dataset.collate_fn)

date_range = pd.date_range((1990, 1, 1), (1995, 1, 1))
start_date_ig = (1994, 6, 1)
end_date_ig = (1995, 6, 15)
new_date_range, idx =  get_date_range_and_idx(start_date_ig, end_date_ig, date_range)

ig =  IntegratedGradients(cuda_lstm, multiply_by_inputs=True)
basline = torch.zeros(dataset._x_d[idx[0]:idx[0]+1, : , : ].shape)
integ_grad = np.zeros(dataset._x_d[idx[0]:idx[0]+1, : , : ].shape)


#
# path_to_model = ''
# # ds_val - need to see what to do with it. I think it's for getting the data out or something like it
#
#
# # set up date ranges and stuff
# start_date_tpl = ds_val.dates[0]
# start_date = pd.to_datetime(datetime.datetime(start_date_tpl[0], start_date_tpl[1], start_date_tpl[2], 0, 0)) + pd.DateOffset(days=ds_val.seq_length + ds_val.lead)
# end_date_tpl = ds_val.dates[1]
# temp = pd.to_datetime(datetime.datetime(end_date_tpl[0], end_date_tpl[1], end_date_tpl[2], 0, 0))
# end_date = temp + pd.DateOffset(days=1)
# date_range = pd.date_range(start_date, end_date)
#
# # load model
# model = torch.load(path_to_model)
# model.eval()
# # Calculate Integrated Gradients
# start_date_ig = (1994, 6, 1)
# end_date_ig = (1995, 6, 15)
# new_date_range, idx =  get_date_range_and_idx(start_date_ig, end_date_ig, date_range)
# # set model to eval mode (important for dropout)
# model.eval()
# ig =  IntegratedGradients(model, multiply_by_inputs=True)
# basline = torch.zeros(ds_val.x[idx[0]:idx[0]+1, : , : ].shape)
# integ_grad = np.zeros(ds_val.x[idx[0]:idx[0]+1, : , : ].shape)
# for i in idx:
#   integ_grad += ig.attribute(ds_val.x[i:(i+1), : , : ], basline).numpy()
# integ_grad = np.squeeze(integ_grad)
# integ_grad /= len(idx)
# _ = model.cuda()