import torch
import os
BASELINE_NON_ZERO_VALUES = {'temperature_2m_mean': 'tmp_dc_syr'}
ATTRIBUTES_DIR = r'C:\Users\omer6\Documents\Research\Caravan\attributes\camels'


def create_baseline(list_of_inputs, seq_length, basin_file):
    with open(basin_file) as f:
        basin_name = f.readline()[:-1]
    basline = torch.zeros([1, seq_length, len(list_of_inputs)])
    for i, parameter in enumerate(list_of_inputs):
        if parameter in BASELINE_NON_ZERO_VALUES.keys():
            attribute_value = float(find_attribute_value(basin_name, BASELINE_NON_ZERO_VALUES[parameter]))
            if BASELINE_NON_ZERO_VALUES[parameter] == 'tmp_dc_syr':
                attribute_value /= 10
            for j in range(seq_length):
                basline[0, j, i] = attribute_value
    return basline


def find_attribute_value(basin_name, attribute):
    file_list = os.listdir(ATTRIBUTES_DIR)
    attribute_value = None
    for file in file_list:
        with open(ATTRIBUTES_DIR + '/' + file) as f:
            atts_in_file = f.readline().split(',')
            if attribute in atts_in_file:
                att_index = atts_in_file.index(attribute)
                for row in f:
                    row_values = row.split(',')
                    if row_values[0] == basin_name:
                        attribute_value = row_values[att_index]
                        break
    return attribute_value
