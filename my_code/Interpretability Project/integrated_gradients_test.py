from captum.attr import IntegratedGradients
import torch
import numpy as np
import datetime
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuralhydrology.datasetzoo import get_dataset, camelsus
from neuralhydrology.evaluation.utils import load_scaler
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.utils.config import Config

from IntegLSTM import IntegLSTM
from integ_utils import create_baseline

WANTED_DATES = ['1993-04-12'] #, '1993-12-12', '1988-03-28']  # make sure the test dates include them
LOAD_FROM_FILE = False
DAYS_BEFORE = 14
CONFIG_FILE_PATH = r'..\Configs\integrated_gradients_test1.yml'
RUN_DIR_PATH = r'..\runs\60_train_benchmark'
GRAPH_PARAMETERS = ['total_precipitation_sum', 'potential_evaporation_sum','snow_depth_water_equivalent_mean', 'surface_net_solar_radiation_mean', 'temperature_2m_mean']
ALL_PARAMETERS = ['p_mean', 'pet_mean', 'aridity', 'frac_snow', 'high_prec_freq', 'high_prec_dur',
                  'low_prec_freq', 'low_prec_dur', 'moisture_index', 'seasonality',
                  'snow_depth_water_equivalent_mean', 'surface_net_solar_radiation_mean',
                  'surface_net_thermal_radiation_mean', 'surface_pressure_mean', 'temperature_2m_mean',
                  'dewpoint_temperature_2m_mean', 'u_component_of_wind_10m_mean', 'v_component_of_wind_10m_mean',
                  'volumetric_soil_water_layer_1_mean', 'volumetric_soil_water_layer_2_mean',
                  'volumetric_soil_water_layer_3_mean', 'volumetric_soil_water_layer_4_mean',
                  'total_precipitation_sum', 'potential_evaporation_sum']
# todo: get parameters from config file

config_file = Path(CONFIG_FILE_PATH)
cudalstm_config = Config(config_file)
run_dir = Path(RUN_DIR_PATH)  # maybe get from config?

if LOAD_FROM_FILE is False:

    # Load model with random weights
    cuda_lstm = CudaLSTM(cfg=cudalstm_config)
    integ_lstm = IntegLSTM(cfg=cudalstm_config)
    # load the trained weights into the new model.
    model_path = run_dir / 'model_epoch006.pt'
    model_weights = torch.load(str(model_path), map_location='cuda:0')  # load the weights from the file, creating the weight tensors on CPU
    cuda_lstm.load_state_dict(model_weights)  # set the new model's weights to the values loaded from file

    integ_lstm.copy_weights(cuda_lstm)

    integ_lstm.eval()

    # load the dataset
    scaler = load_scaler(run_dir)
    dataset = get_dataset(cudalstm_config, is_train=False, period='test', scaler=scaler)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    ig = IntegratedGradients(integ_lstm, multiply_by_inputs=True)

    static_attributes = integ_lstm.cfg.static_attributes
    dynamic_inputs = integ_lstm.cfg.dynamic_inputs

    input_size = len(dynamic_inputs) + len(static_attributes)

    #basline = torch.zeros([1, integ_lstm.cfg.seq_length, input_size])
    basline = create_baseline(ALL_PARAMETERS, integ_lstm.cfg.seq_length, scaler)
    integ_grad = np.zeros([1, integ_lstm.cfg.seq_length, input_size])

    cnt_wanted_dates = 0
    # todo: find out what is scaler and how it affects the results, something here is off when creating different baseline
    for sample in dataloader:
        if str(sample['date'][0][-1])[:10] in WANTED_DATES:
            cnt_wanted_dates += 1
            seq_static = sample['x_s'].expand(sample['x_s'].size(0), integ_lstm.cfg.seq_length, sample['x_s'].size(1))
            input = torch.cat((seq_static, sample['x_d']), 2)
            integ_grad += ig.attribute(input, basline).numpy()

    if cnt_wanted_dates < len(WANTED_DATES):
        print("ERROR: Couldn't use all wanted dates")

    integ_grad = np.squeeze(integ_grad)
    integ_grad /= cnt_wanted_dates
    torch.save(integ_grad, 'integ_grad_result_single_date.pt')

else:
    integ_grad = torch.load('integ_grad_result_.pt')

integ_grad_sliced = integ_grad[-DAYS_BEFORE:, :]

set_graph_parameters = set(GRAPH_PARAMETERS)
graph_parameters_indices = [ALL_PARAMETERS.index(x) for x in GRAPH_PARAMETERS if x in ALL_PARAMETERS]


plt.plot(np.arange(-integ_grad_sliced.shape[0], 0)+1, integ_grad_sliced[:, graph_parameters_indices], marker='o')
plt.xlabel('Day')
plt.ylabel('Integrated Gradients')
plt.legend(GRAPH_PARAMETERS, loc='upper left')
plt.grid()
plt.show()

# List of inputs:



