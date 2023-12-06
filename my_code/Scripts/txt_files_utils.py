import os
# write_file = open('482_basin_caravan_list.txt', 'w')
# read_file = open('531_basin_list.txt', 'r')
# for line in read_file:
#     write_file.write('camels_'+line)
# write_file.close()
# read_file.close()

AREA_BIGGER_THAN = 500
CHECK_AREA = False
ATT_FILE_PATH = r"C:\Users\omer6\Documents\Research\Caravan\attributes\hysets\attributes_other_hysets.csv"
BASIN_FILE_NAME = 'caravan_lamah_basins.txt'
TIMESERIES_DIR_PATH = '../../../Caravan/timeseries/netcdf/lamah'
def create_basin_file_from_dir():
    write_file = open('../BasinFiles/'+BASIN_FILE_NAME, 'w')
    existing_basins = []
    for subdir, dirs, files in os.walk(TIMESERIES_DIR_PATH):
        for file in files:
            basin_name = file.split('.')[0]
            basin_number = basin_name.split('_')[1]
            if basin_number not in existing_basins:
                existing_basins.append(basin_number)
                if CHECK_AREA is True:
                    with open(ATT_FILE_PATH) as f:
                        atts_in_file = f.readline().split(',')
                        if 'area\n' not in atts_in_file:
                            print('ERROR NO AREA IN FILE')
                        for row in f:
                            row_values = row.split(',')
                            if row_values[0] == basin_name:
                                area_value = row_values[-1]
                                if float(area_value) >= AREA_BIGGER_THAN:
                                    write_file.write(file.split('.')[0]+'\n')
                else:
                    write_file.write(basin_name + '\n')
    write_file.close()


def find_max_date_of_data_in_dir():
    data_dir = '../../../Caravan/timeseries/csv/camels'
    file_list = os.listdir(data_dir)
    for file in file_list:
        with open(data_dir + '/' + file) as f:
            last_column = [row.split(',')[-1] for row in f]
        with open(data_dir + '/' + file) as f:
            first_column = [row.split(',')[0] for row in f]
        cnt_empty_elements = 0
        for i, element in enumerate(last_column):
            if element == '\n':
                is_end = True
                for j in range(i,len(last_column)):
                    if last_column[j] != '\n':
                        is_end = False
                        break
                if is_end is True:
                    print(file, first_column[i])
                    break

create_basin_file_from_dir()