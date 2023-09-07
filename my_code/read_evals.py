import pandas as pd
from pathlib import Path
from graphs_utils import plot_nse_cdf

run_dir = Path(r'C:\Users\omer6\Documents\Research\neuralhydrology\my_code\runs\60_train_cluster_half_static_0509_192336')
output_filer_path = run_dir / "test/model_epoch007/test_metrics.csv"
# load test results of the base run
df_final = pd.read_csv(output_filer_path, dtype={'basin': str})
df_final = df_final.set_index("basin")

print(f"Median NSE of the test period {df_final['NSE'].median():.3f}")

plot_nse_cdf(output_filer_path, r'C:\Users\omer6\Documents\Research\neuralhydrology\my_code\runs\60_train_cluster_half_static_0509_192336\NSE_CDF_graph_epoch7.png')
