import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--runsdir', type=str, required=True)
args = parser.parse_args()


runs_dir = args.runsdir
for_tensor_dir = runs_dir + r'/for_tensor'
os.mkdir(for_tensor_dir)

for dir in os.listdir(runs_dir):
    if '.' not in dir and dir != 'for_tensor' and dir != '_old':
        files = os.listdir(runs_dir + "/" + dir)
        os.mkdir(for_tensor_dir + '/' + dir)
        for file in files:
            if 'event' in file or 'output' in file:
                shutil.copy(runs_dir + "/" + dir + '/' + file, for_tensor_dir + '/' + dir)


