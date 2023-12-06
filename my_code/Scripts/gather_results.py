import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--runsdir', type=str, required=True)
parser.add_argument('--dirprefix', type=str, required=True)
parser.add_argument('--outputdir', type=str, required=True)
parser.add_argument('--runname', type=str, required=True)

args = parser.parse_args()


runs_dir = args.runsdir
dir_prefix = args.dirprefix
run_name = args.runname
output_dir = args.outputdir
os.mkdir(output_dir + "/" + run_name)

cnt_runs = 1
for dir in os.listdir(runs_dir):
    if dir_prefix in dir:
        for root, dirs, files in os.walk(runs_dir + "/" + dir + "/" + 'test'):
            for name in files:
                if '.csv' in name:
                    results_file_path = os.path.join(root, name)
                    shutil.copyfile(results_file_path, output_dir + "/" + run_name + "/" + run_name + str(cnt_runs) + '.csv')
                    cnt_runs += 1


