import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run

# # by default we assume that you have at least one CUDA-capable NVIDIA GPU
# if torch.cuda.is_available():
#     start_run(config_file=Path("531_basins.yml"))

run_dir = Path(r"C:\Users\omer6\Documents\Research\neuralhydrology\my_code\runs\60_train_cluster_half_static_0509_192336")
eval_run(run_dir=run_dir, period="test", epoch=7)
with open(run_dir / "test" / "model_epoch007" / "test_results.p", "rb") as fp:
    results = pickle.load(fp)

results.keys()
# results['01022500']['1D']['xr']

# extract observations and simulations
qobs = results['camels_01022500']['1D']['xr']['QObs(mm/d)_obs']
qsim = results['camels_01022500']['1D']['xr']['QObs(mm/d)_sim']

fig, ax = plt.subplots(figsize=(16,10))
ax.plot(qobs['date'], qobs)
ax.plot(qsim['date'], qsim)
ax.set_ylabel("Discharge (mm/d)")
ax.set_title(f"Test period - NSE {results['01022500']['1D']['NSE']:.3f}")

values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))
for key, val in values.items():
    print(f"{key}: {val:.3f}")