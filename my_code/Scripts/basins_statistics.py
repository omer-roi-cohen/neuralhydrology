

def cnt_basins_per_area():
    basins_file_path = r'C:\Users\omer6\Documents\Research\neuralhydrology\my_code\BasinFiles\caravan_filtered_05_NSE.txt'
    f = open(basins_file_path, 'r')
    camelsaus_cnt = 0
    camels_cnt = 0
    camelsbr_cnt = 0
    camelscl_cnt = 0
    camelsgb_cnt = 0
    hysets_cnt = 0
    lamah_cnt = 0
    for line in f.readlines():
        if 'camelsaus' in line:
            camelsaus_cnt += 1
        elif 'camelsbr' in line:
            camelsbr_cnt += 1
        elif 'camelscl' in line:
            camelscl_cnt += 1
        elif 'camelsgb' in line:
            camelsgb_cnt += 1
        elif 'hysets' in line:
            hysets_cnt += 1
        elif 'lamah' in line:
            lamah_cnt += 1
        elif 'camels' in line:
            camels_cnt += 1
    print('Basins Count:')
    print('CAMELS (US):', camels_cnt, '('+str(int((camels_cnt/482)*100))+'%)')
    print('CAMELS-AUS:', camelsaus_cnt, '('+str(int((camelsaus_cnt/150)*100))+'%)')
    print('CAMELS-BR:', camelsbr_cnt, '('+str(int((camelsbr_cnt/376)*100))+'%)')
    print('CAMELS-CL:', camelscl_cnt, '('+str(int((camelscl_cnt/314)*100))+'%)')
    print('CAMELS-GB:', camelsgb_cnt, '('+str(int((camelsgb_cnt/408)*100))+'%)')
    print('HYSETS:', hysets_cnt, '('+str(int((hysets_cnt/4166)*100))+'%)')
    print('LamaH-CE:', lamah_cnt, '('+str(int((lamah_cnt/479)*100))+'%)')

cnt_basins_per_area()