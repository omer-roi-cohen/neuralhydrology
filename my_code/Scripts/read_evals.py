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
    output_filer_path = run_dir / "test/model_epoch006/test_metrics.csv"
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


def multi_read_eval():
    chosen_tests= ['finetune_all_model.csv', 'finetune_only_head.csv', 'us.csv']
    metrics_dir = Path(r'C:\Users\omer6\Documents\Research\neuralhydrology\my_code\BatchOutput')
    for filename in os.listdir(metrics_dir):
        f = os.path.join(metrics_dir, filename)
        if str(f).endswith(".csv") and str(filename) in chosen_tests:
            metric_df = pd.read_csv(str(f), dtype={'basin': str})
            metric_df = metric_df.set_index("basin")
            nse_vec = metric_df['NSE'].to_numpy()
            x, y = ecdf(nse_vec[nse_vec > 0])
            x = np.insert(x, 0, x[0])
            y = np.insert(y, 0, 0.)
            legend_label = str(filename).split('.')[0] + f" Median: {metric_df['NSE'].median():.3f}"
            plt.plot(x, y, label=legend_label)
            print(filename)
            print(f"Median NSE of the test period {metric_df['NSE'].median():.3f}")
    plt.ylabel('CDF')
    plt.xlabel('NSE')
    plt.legend()
    plt.show()

multi_read_eval()