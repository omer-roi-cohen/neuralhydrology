import os
from pathlib import Path
from statistics import mean
from tqdm import tqdm
import pandas as pd

def filter_basins_by_data_size():
    train_start_date = "2001-01-01"
    train_end_date = "2020-12-31"
    val_start_date = "1981-01-01"
    val_end_date = "1990-12-31"
    test_start_date = "1991-01-01"
    test_end_date = "2000-12-31"

    min_train_yrs = 10
    min_val_yrs = 5
    min_test_yrs = 5

    basin_file_name = 'caravan_min_data.txt'
    basins_folder = Path(r"C:\Users\omer6\Documents\Research\Caravan\timeseries\csv")
    write_file = open('../BasinFiles/'+basin_file_name, 'w')
    train_yrs_list = []
    val_yrs_list = []
    test_yrs_list = []
    existing_basins = []
    for subdir, dirs, files in os.walk(basins_folder):
        for file in tqdm(files):
            basin_name = file.split('.')[0]
            basin_number = basin_name.split('_')[1]
            if basin_number not in existing_basins:
                existing_basins.append(basin_number)
                basin_file_path = subdir + '\\' + file
                train_yrs = num_of_yrs_in_basin_in_time_period(basin_file_path, train_start_date, train_end_date)
                train_yrs_list.append(train_yrs)
                val_yrs = num_of_yrs_in_basin_in_time_period(basin_file_path, val_start_date, val_end_date)
                val_yrs_list.append(val_yrs)
                test_yrs = num_of_yrs_in_basin_in_time_period(basin_file_path, test_start_date, test_end_date)
                test_yrs_list.append(test_yrs)
                if train_yrs >= min_train_yrs and val_yrs >= min_val_yrs and test_yrs >= min_test_yrs:
                    write_file.write(basin_name + '\n')
    print(mean(train_yrs_list))
    print(mean(val_yrs_list))
    print(mean(test_yrs_list))


def num_of_yrs_in_basin_in_time_period(basin_file_path, date_start, date_end):
    with open(basin_file_path) as f:
        last_column = [row.split(',')[-1] for row in f]
    with open(basin_file_path) as f:
        first_column = [row.split(',')[0] for row in f]
    cnt_data_elements = 0
    first_column = first_column[1:]
    last_column = last_column[1:]
    for i, element in enumerate(last_column):
        if date_end >= first_column[i] >= date_start and element != '\n' and i >= 364:
            curr_date = first_column[i]
            cnt_data_elements += 1
    return cnt_data_elements / 365.0


def filter_results_file_by_basinfile():
    metrics_file_path = r'C:\Users\omer6\Documents\Research\neuralhydrology\my_code\runs\all_data_caravan_one_layer_1511_100346\test\model_epoch015\test_metrics.csv'
    basin_file_path = r'C:\Users\omer6\Documents\Research\neuralhydrology\my_code\BasinFiles\caravan_filtered_05_NSE.txt'
    output_path = r'C:\Users\omer6\Documents\Research\neuralhydrology\my_code\BatchOutput\caravan_filterred_05_NSE.csv'
    input_file = open(basin_file_path, 'r')
    basins_list = input_file.read().splitlines()
    metric_df = pd.read_csv(metrics_file_path, dtype={'basin': str})
    metric_df = metric_df.loc[metric_df['basin'].isin(basins_list)]
    metric_df.to_csv(output_path)

def filter_basins_by_validation_score():
    input_basin_file_to_filter = r'C:\Users\omer6\Documents\Research\neuralhydrology\my_code\BasinFiles\caravan_lamah_basins.txt'
    validation_score_csv_path = r'C:\Users\omer6\Documents\Research\neuralhydrology\my_code\runs\essenmble_results\all_data\validation_ensemble_metrics.csv'
    validation_NSE_threshold = 0.5
    basin_file_name = 'carave_filtered_05_NSE.txt'
    write_file = open('../BasinFiles/'+basin_file_name, 'w')
    input_file = open(input_basin_file_to_filter, 'r')
    basins_list = input_file.read().splitlines()

    metric_df = pd.read_csv(validation_score_csv_path, dtype={'basin': str})
    metric_df = metric_df.loc[metric_df['basin'].isin(basins_list)]
    nse_vec = metric_df['NSE'].to_numpy()
    basin_vec = metric_df['basin'].to_numpy()
    for i, basin in enumerate(tqdm(basin_vec)):
        if str(nse_vec[i]) != 'nan':
            if float(nse_vec[i]) >= validation_NSE_threshold:
                write_file.write(basin + '\n')


filter_results_file_by_basinfile()
#filter_basins_by_validation_score()
#num = num_of_yrs_in_basin_in_time_period(r'C:\Users\omer6\Documents\Research\Caravan\timeseries\csv\camelscl\camelscl_1001002.csv', "1981-01-01", "1990-12-31")
#print(num)