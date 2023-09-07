import os
# write_file = open('482_basin_caravan_list.txt', 'w')
# read_file = open('531_basin_list.txt', 'r')
# for line in read_file:
#     write_file.write('camels_'+line)
# write_file.close()
# read_file.close()
file_list = os.listdir('../data/Caravan/timeseries/netcdf/camels')
print(file_list)
write_file = open('482_basin_caravan_list.txt', 'w')
for file in file_list:
    write_file.write(file.split('.')[0]+'\n')
write_file.close()
