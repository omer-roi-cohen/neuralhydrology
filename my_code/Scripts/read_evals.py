import pandas as pd
from pathlib import Path
from graphs_utils import plot_nse_cdf, ecdf
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from tqdm import tqdm

def nse_in_relations_to_area():
    metrics_csv = r'C:\Users\omer6\Documents\Research\neuralhydrology\my_code\BatchOutput\hysets_epoch11.csv'
    att_file_path = r"C:\Users\omer6\Documents\Research\Caravan\attributes\hysets\attributes_other_hysets.csv"
    metric_df = pd.read_csv(metrics_csv, dtype={'basin': str})
    nse_vec = metric_df['NSE'].to_numpy()
    basin_vec = metric_df['basin'].to_numpy()
    x_vec = []
    y_vec = []
    for i, basin in enumerate(tqdm(basin_vec)):
        if str(nse_vec[i]) != 'nan':
            with open(att_file_path) as f:
                atts_in_file = f.readline().split(',')
                if 'area\n' not in atts_in_file:
                    print('ERROR NO AREA IN FILE')
                for row in f:
                    row_values = row.split(',')
                    if row_values[0] == basin:
                        area_value = float(row_values[-1])
                        x_vec.append(area_value)
                        y_vec.append(nse_vec[i])
    plt.scatter(x_vec, y_vec, alpha=0.2, s=100)
    plt.ylim([0, 1])
    plt.xlabel('Area - Km^2')
    plt.ylabel('NSE')
    plt.show()

def single_read_eval():
    run_dir = Path(r'../runs/60_train_cluster_half_static_0509_192336')
    output_filer_path = run_dir / "test/model_epoch006/one_layer_20_dropout06.csv"
    # load test results of the base run
    df_final = pd.read_csv(output_filer_path, dtype={'basin': str})
    df_final = df_final.set_index("basin")

    print(f"Median NSE of the test period {df_final['NSE'].median():.3f}")

    plot_nse_cdf(output_filer_path, run_dir / 'NSE_CDF_graph_epoch6.png')

    with open(run_dir / "test" / "model_epoch006" / "test_results.p", "rb") as fp:
        results = pickle.load(fp)

    results.keys()
    # results['01022500']['1D']['xr']

    # extract observations and simulations
    qobs = results['camels_01022500']['1D']['xr']['streamflow_obs']
    qsim = results['camels_01022500']['1D']['xr']['streamflow_sim']

    fig, ax = plt.subplots(figsize=(16,10))
    ax.plot(qobs['date'], qobs)
    ax.plot(qsim['date'], qsim)
    ax.set_ylabel("Discharge (mm/d)")
    ax.set_title(f"Test period - NSE {results['camels_01022500']['1D']['NSE']:.3f}")
    plt.show()


def combine_regional_models_results():
    metrics_dir = Path(r'C:\Users\omer6\Documents\Research\neuralhydrology\my_code\BatchOutput')
    chosen_tests = ['camels.csv', 'camelsaus.csv', 'camelsbr.csv', 'camelscl.csv', 'camelsgb.csv', 'hysets.csv', 'lamah.csv']
    camels_basin_file_path = r'C:\Users\omer6\Documents\Research\neuralhydrology\my_code\BasinFiles\caravan_camels_basins.txt'
    output_file_path = r'/my_code/BatchOutput/regional_models.csv'
    camels_basin_file = open(camels_basin_file_path, 'r')
    camels_basins_list = camels_basin_file.read().splitlines()
    camels_basins_list = [str(i).removeprefix('camels_') for i in camels_basins_list]
    df_vector = []
    for filename in os.listdir(metrics_dir):
        f = os.path.join(metrics_dir, filename)
        if str(f).endswith(".csv") and str(filename) in chosen_tests:
            file_metric_df = pd.read_csv(str(f), dtype={'basin': str})
            # if 'hysets' in str(filename):
            #     hysets_basin_names = file_metric_df['basin']
            #     hysets_basin_numbers = hysets_basin_names.str.removeprefix('hysets_')
            #     file_metric_df = file_metric_df.loc[~hysets_basin_numbers.isin(camels_basins_list)]
            df_vector.append(file_metric_df)
    all_df = pd.concat(df_vector)
    all_df.to_csv(output_file_path)

def multi_read_eval_ensemble():
    chosen_metric = 'NSE'
    metrics_dir = Path(r'C:\Users\omer6\Documents\Research\neuralhydrology\my_code\BatchOutput')
    basin_file_to_check = r'C:\Users\omer6\Documents\Research\neuralhydrology\my_code\BasinFiles\caravan_camels_basins.txt'
    chosen_tests_folders = ['camels', 'caravan']

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    basin_file = open(basin_file_to_check, 'r')
    basins_list = basin_file.read().splitlines()

    for i, chosen_test in enumerate(chosen_tests_folders):
        chosen_test_path = os.path.join(metrics_dir, chosen_test)
        for filename in os.listdir(chosen_test_path):
            f = os.path.join(chosen_test_path, filename)
            if str(f).endswith(".csv"):
                metric_df = pd.read_csv(str(f), dtype={'basin': str})
                metric_df = metric_df.loc[metric_df['basin'].isin(basins_list)]
                metric_df = metric_df.set_index("basin")
                if chosen_metric == 'NSE':
                    nse_vec = metric_df['NSE'].to_numpy()
                    x, y = ecdf(nse_vec[nse_vec > 0])
                    x = np.insert(x, 0, x[0])
                    y = np.insert(y, 0, 0.)
                if chosen_metric == 'FHV':
                    fhv_vec = metric_df['FHV'].to_numpy()
                    fhv_vec = fhv_vec[~np.isnan(fhv_vec)]
                    fhv_vec_abs = np.absolute(fhv_vec)
                    x, y = ecdf(fhv_vec_abs[fhv_vec_abs < 100])
                    x = np.insert(x, 0, x[0])
                    y = np.insert(y, 0, 0.)
                    plt.xlim([0, 80])
                if chosen_metric == 'KGE':
                    kge_vec = metric_df['KGE'].to_numpy()
                    x, y = ecdf(kge_vec[kge_vec > 0])
                    x = np.insert(x, 0, x[0])
                    y = np.insert(y, 0, 0.)
                if 'ensemble' in str(f):
                    legend_label = str(filename).split('.')[0] + f" Median: {metric_df[chosen_metric].median():.3f}"
                    markevery = int(len(x) / 10) + i* 20
                    plt.plot(x, y, label=legend_label, color=color_cycle[i], alpha=0.95,  linestyle='--', marker='^', markevery=markevery)
                else:
                    plt.plot(x, y, color=color_cycle[i], alpha=0.2)
                print(filename)
                print(f"Median of the test period {metric_df['NSE'].median():.3f}")
                plt.xlabel(chosen_metric)
    plt.ylabel('CDF')
    plt.legend()
    plt.show()

def multi_read_eval():
    chosen_metric = 'FHV'
    metrics_dir = Path(r'C:\Users\omer6\Documents\Research\neuralhydrology\my_code\BatchOutput')
    basin_file_to_check = r'C:\Users\omer6\Documents\Research\neuralhydrology\my_code\BasinFiles\caravan_all_basins.txt'
    chosen_tests = ['generalized_model.csv', 'generalized_model_ensemble.csv', 'regional_models.csv', 'regional_models_ensemble.csv']
    basin_file = open(basin_file_to_check, 'r')
    basins_list = basin_file.read().splitlines()
    for filename in os.listdir(metrics_dir):
        f = os.path.join(metrics_dir, filename)
        if str(f).endswith(".csv") and str(filename) in chosen_tests:
            metric_df = pd.read_csv(str(f), dtype={'basin': str})
            metric_df = metric_df.loc[metric_df['basin'].isin(basins_list)]
            metric_df = metric_df.set_index("basin")
            if chosen_metric == 'NSE':
                nse_vec = metric_df['NSE'].to_numpy()
                x, y = ecdf(nse_vec[nse_vec > 0])
                x = np.insert(x, 0, x[0])
                y = np.insert(y, 0, 0.)
                legend_label = str(filename).split('.')[0] + f" Median: {metric_df['NSE'].median():.3f}"
                plt.plot(x, y, label=legend_label)
                print(filename)
                print(f"Median NSE of the test period {metric_df['NSE'].median():.3f}")
                plt.xlabel('NSE')
            if chosen_metric =='FHV':
                fhv_vec = metric_df['FHV'].to_numpy()
                fhv_vec = fhv_vec[~np.isnan(fhv_vec)]
                fhv_vec_abs = np.absolute(fhv_vec)
                x, y = ecdf(fhv_vec_abs[fhv_vec_abs < 100])
                x = np.insert(x, 0, x[0])
                y = np.insert(y, 0, 0.)
                legend_label = str(filename).split('.')[0] + f" Median: {np.median(fhv_vec_abs):.3f}"
                plt.plot(x, y, label=legend_label)
                plt.xlim([0, 80])
                print(filename)
                print(f"Median FHV of the test period {np.median(fhv_vec_abs):.3f}")
                plt.xlabel('FHV')
            if chosen_metric == 'KGE':
                kge_vec = metric_df['KGE'].to_numpy()
                x, y = ecdf(kge_vec[kge_vec > 0])
                x = np.insert(x, 0, x[0])
                y = np.insert(y, 0, 0.)
                legend_label = str(filename).split('.')[0] + f" Median: {metric_df['KGE'].median():.3f}"
                plt.plot(x, y, label=legend_label)
                print(filename)
                print(f"Median KGE of the test period {metric_df['KGE'].median():.3f}")
                plt.xlabel('KGE')
    plt.ylabel('CDF')
    plt.legend()
    plt.show()

multi_read_eval_ensemble()
#multi_read_eval()
#combine_regional_models_results()