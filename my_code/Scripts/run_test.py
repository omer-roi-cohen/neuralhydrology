import sys
from pathlib import Path
sys.path.append("../../neuralhydrology/")
sys.path.append("../../")
from neuralhydrology.nh_run import eval_run

sys.path.append("../../neuralhydrology/")


# # by default we assume that you have at least one CUDA-capable NVIDIA GPU
# if torch.cuda.is_available():
#     start_run(config_file=Path("531_basins.yml"))

# run_dir = Path(r"runs/60_train_clip_gradients_2")
# eval_run(run_dir=run_dir, period="test", epoch=5)
#
# run_dir = Path(r"runs/60_train_clip_gradients_5")
# eval_run(run_dir=run_dir, period="test", epoch=9)

run_dir = Path(r"runs/60_train_all_data_2009_213617")
eval_run(run_dir=run_dir, period="test", epoch=14)

run_dir = Path(r"runs/finetune_hysets_on_camels_all_2109_144620")
eval_run(run_dir=run_dir, period="test", epoch=1)

run_dir = Path(r"runs/finetune_hysets_on_camels_head_2109_162206")
eval_run(run_dir=run_dir, period="test", epoch=3)

run_dir = Path(r"runs/60_train_hysets_area500_2109_001143")
eval_run(run_dir=run_dir, period="test", epoch=8)
